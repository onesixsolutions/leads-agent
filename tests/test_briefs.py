"""
Tests for lead briefs: rendering, the S3 key/version scheme, URL routing and
the "never break lead processing" contract.

All offline. There is no boto3 client and no AWS credentials anywhere in here:
the store takes an injected client, so a dict-backed stub is the whole test
double.
"""

from __future__ import annotations

import json
from html import escape as html_escape

import pytest

from leads_agent.briefs import (
    BriefRef,
    BriefStore,
    brief_path,
    history_path,
    lead_id_for,
    publish_brief,
    render_brief_html,
    render_history_html,
    resolve_route,
)
from leads_agent.briefs.identity import is_valid_lead_id
from leads_agent.briefs.routes import RouteKind
from leads_agent.briefs.server import handle_route
from leads_agent.config import Settings
from leads_agent.icp_fit import CRITERIA, apply_icp_fit
from leads_agent.models import (
    CompanyResearch,
    ContactResearch,
    CriterionStatus,
    EnrichedLeadClassification,
    HubSpotLead,
    ICPAssessment,
    ICPCriterion,
    ICPVerdict,
    LeadLabel,
    OutreachBrief,
)

BUCKET = "test-briefs"


# --- Test doubles ---------------------------------------------------------


class NotFound(Exception):
    """Stands in for botocore's ClientError with a 404-shaped response."""

    def __init__(self, code: str = "NoSuchKey") -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    """In-memory stand-in for the handful of S3 calls the store makes."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[str] = []
        self.fail_put_with: Exception | None = None
        self.fail_get_with: Exception | None = None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> dict:
        if self.fail_put_with is not None:
            raise self.fail_put_with
        assert Bucket == BUCKET
        self.objects[Key] = Body
        self.put_calls.append(Key)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if self.fail_get_with is not None:
            raise self.fail_get_with
        assert Bucket == BUCKET
        if Key not in self.objects:
            raise NotFound()
        return {"Body": _Body(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == BUCKET
        if Key not in self.objects:
            raise NotFound("404")
        return {"ContentLength": len(self.objects[Key])}


def make_settings(**overrides: object) -> Settings:
    """Settings with briefs configured, overridable per test."""
    values: dict[str, object] = {
        "BRIEFS_ENABLED": True,
        "BRIEFS_S3_BUCKET": BUCKET,
        "BRIEFS_S3_PREFIX": "briefs",
        "BRIEFS_BASE_URL": "http://100.79.160.6:8080",
        "BRIEFS_HTTP_ENABLED": False,
    }
    values.update(overrides)
    return Settings(**values)


# --- Fixtures ------------------------------------------------------------


def make_lead(**overrides: object) -> HubSpotLead:
    values: dict[str, object] = {
        "first_name": "Dana",
        "last_name": "Okafor",
        "email": "dana.okafor@northgate-mfg.com",
        "company": "Northgate Manufacturing",
        "message": "We need help migrating our warehouse to Snowflake.",
        "raw_text": "*First Name*: Dana",
    }
    values.update(overrides)
    return HubSpotLead(**values)


def make_assessment(default: CriterionStatus = CriterionStatus.met, **overrides) -> ICPAssessment:
    fields = {
        spec.field: ICPCriterion(
            status=default,
            finding=f"{spec.label} determined",
            evidence=None if default == CriterionStatus.unknown else "corroborated in search",
        )
        for spec in CRITERIA
    }
    for name, status in overrides.items():
        fields[name] = ICPCriterion(
            status=status,
            finding=f"{name} is {status.value}",
            evidence=None if status == CriterionStatus.unknown else "evidence",
        )
    return ICPAssessment(**fields)


def make_classification(**overrides) -> EnrichedLeadClassification:
    """A fully-populated enriched classification, built by hand (no LLM)."""
    values: dict[str, object] = {
        "first_name": "Dana",
        "last_name": "Okafor",
        "email": "dana.okafor@northgate-mfg.com",
        "company": "Northgate Manufacturing",
        "label": LeadLabel.promising,
        "reason": "Named exec at a mid-market manufacturer with a migration mandate.",
        "lead_summary": "VP Data wants Snowflake migration help.",
        "key_signals": ["budget mentioned", "named platform"],
        "company_research": CompanyResearch(
            company_name="Northgate Manufacturing",
            company_description="Tier-two automotive supplier.\n\nRuns 11 plants.",
            industry="Manufacturing",
            company_size="$1.4B revenue, 4,200 staff",
            website="northgate-mfg.com",
            relevance_notes="Focus overlay: Manufacturing.",
        ),
        "contact_research": ContactResearch(
            full_name="Dana Okafor",
            title="VP, Data & Analytics",
            linkedin_summary="Eight years at Northgate.",
            relevance_notes="Owns the platform budget.",
        ),
        "research_summary": "Credible mid-market buyer with an active mandate.",
        "icp_assessment": make_assessment(),
        "brief": OutreachBrief(
            icp_statement="$1.4B manufacturer migrating to Snowflake, Modernization door.",
            analyst_take="Reads like a real programme. Move on it.",
            opportunity="Plant-level rollout after the first migration.",
            risks=["Procurement cycle may be slow"],
            exception_case=None,
            recommended_entry="Modernization & Migration, $250-400k first phase.",
            talking_points=["They named Snowflake themselves", "11 plants means phase two"],
            accelerators=["Migration Accelerator"],
        ),
    }
    values.update(overrides)
    return apply_icp_fit(EnrichedLeadClassification(**values))


# --- Identity ------------------------------------------------------------


def test_same_email_gives_same_lead_id_across_runs():
    """Re-analysis must land under the same id, or history cannot accumulate."""
    first = make_lead(message="First enquiry")
    later = make_lead(message="Second enquiry, weeks later", company="Northgate Mfg Inc")
    assert lead_id_for(first) == lead_id_for(later)


def test_different_leads_get_different_ids():
    assert lead_id_for(make_lead()) != lead_id_for(make_lead(email="other@elsewhere.com"))


def test_lead_id_is_slug_plus_digest_and_url_safe():
    lead_id = lead_id_for(make_lead())
    # Slug is the email domain, not the typed company name: the domain cannot
    # drift between enquiries, so the id cannot either.
    assert lead_id.startswith("northgate-mfg-com-")
    assert is_valid_lead_id(lead_id)
    # The individual must not be recoverable from a link pasted into Slack.
    assert "dana" not in lead_id and "@" not in lead_id


def test_lead_with_no_identifying_fields_still_gets_an_id():
    lead_id = lead_id_for(HubSpotLead(raw_text="anonymous blob"))
    assert lead_id.startswith("lead-")
    assert is_valid_lead_id(lead_id)


def test_falls_back_to_name_when_email_is_missing():
    a = lead_id_for(make_lead(email=None))
    b = lead_id_for(make_lead(email=None, message="different message"))
    assert a == b


@pytest.mark.parametrize(
    "bad",
    ["../etc/passwd", "Northgate", "a/b", "-leading", "with space", "", "x" * 200],
)
def test_invalid_lead_ids_are_rejected(bad):
    assert not is_valid_lead_id(bad)


# --- URL scheme ----------------------------------------------------------


def test_brief_and_history_paths():
    assert brief_path("acme-1234567890") == "/briefs/acme-1234567890"
    assert brief_path("acme-1234567890", 3) == "/briefs/acme-1234567890/v3"
    assert history_path("acme-1234567890") == "/briefs/acme-1234567890/history"


@pytest.mark.parametrize(
    ("path", "kind", "version"),
    [
        ("/briefs/acme-1234567890", RouteKind.current_html, None),
        ("/briefs/acme-1234567890/", RouteKind.current_html, None),
        ("/briefs/acme-1234567890/v3", RouteKind.version_html, 3),
        ("/briefs/acme-1234567890/v12?x=1", RouteKind.version_html, 12),
        ("/briefs/acme-1234567890/history", RouteKind.history, None),
        ("/briefs/acme-1234567890.json", RouteKind.current_json, None),
        ("/briefs/acme-1234567890/v2.json", RouteKind.version_json, 2),
    ],
)
def test_route_resolution(path, kind, version):
    route = resolve_route(path)
    assert route is not None
    assert route.kind == kind
    assert route.version == version
    if kind != RouteKind.health:
        assert route.lead_id == "acme-1234567890"


def test_health_route():
    assert resolve_route("/healthz").kind == RouteKind.health


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/briefs",
        "/briefs/",
        "/other/acme-1234567890",
        "/briefs/acme-1234567890/v0",
        "/briefs/acme-1234567890/nope",
        "/briefs/acme-1234567890/v1/extra",
        "/briefs/../secrets",
        "/briefs/Acme-Corp",
    ],
)
def test_unroutable_paths_return_none(path):
    assert resolve_route(path) is None


def test_traversal_cannot_reach_a_key():
    """A path that escapes the prefix must never yield a usable lead_id."""
    assert resolve_route("/briefs/..%2F..%2Fetc/passwd") is None
    assert resolve_route("/briefs/../../etc/passwd") is None


# --- Key scheme and versioning -------------------------------------------


def test_key_scheme_is_zero_padded_so_listings_sort_chronologically():
    store = BriefStore(FakeS3(), BUCKET, "briefs")
    assert store.index_key("acme-1") == "briefs/acme-1/index.json"
    assert store.html_key("acme-1", 1) == "briefs/acme-1/v0001.html"
    assert store.json_key("acme-1", 12) == "briefs/acme-1/v0012.json"
    # Zero padding is what makes a plain `aws s3 ls` chronological.
    assert min(store.html_key("a", 2), store.html_key("a", 10)) == store.html_key("a", 2)


def test_empty_prefix_drops_the_leading_slash():
    store = BriefStore(FakeS3(), BUCKET, "")
    assert store.html_key("acme-1", 1) == "acme-1/v0001.html"


def test_versions_increment_v1_v2_v3():
    client = FakeS3()
    store = BriefStore(client, BUCKET, "briefs")

    for expected in (1, 2, 3):
        version, key = store.publish(
            "acme-1",
            html=f"<p>v{expected}</p>",
            record={"n": expected},
            verdict="in_icp",
            action="follow_up",
        )
        assert version == expected
        assert key == store.html_key("acme-1", expected)

    index = store.read_index("acme-1")
    assert index.current_version == 3
    assert [v.version for v in index.versions] == [1, 2, 3]
    # Earlier versions are still there — that is the whole point.
    assert store.read_html("acme-1", 1) == "<p>v1</p>"
    assert store.read_html("acme-1", 2) == "<p>v2</p>"
    assert store.read_html("acme-1", 3) == "<p>v3</p>"


def test_verdict_change_between_versions_is_recorded():
    """A lead going needs_verification -> in_icp is expected, not an error."""
    store = BriefStore(FakeS3(), BUCKET, "briefs")
    store.publish("acme-1", html="a", record={}, verdict="needs_verification")
    store.publish("acme-1", html="b", record={}, verdict="in_icp")

    index = store.read_index("acme-1")
    assert [v.verdict for v in index.versions] == ["needs_verification", "in_icp"]


def test_structured_record_is_stored_alongside_the_html():
    client = FakeS3()
    store = BriefStore(client, BUCKET, "briefs")
    store.publish("acme-1", html="<p>hi</p>", record={"gate": "revenue_band"})

    raw = store.read_record("acme-1", 1)
    assert json.loads(raw) == {"gate": "revenue_band"}
    assert store.html_key("acme-1", 1) in client.objects
    assert store.json_key("acme-1", 1) in client.objects


def test_lost_index_does_not_overwrite_a_published_brief():
    """The HEAD probe is what makes a missing or corrupt index self-healing."""
    client = FakeS3()
    store = BriefStore(client, BUCKET, "briefs")
    store.publish("acme-1", html="original", record={})

    del client.objects[store.index_key("acme-1")]
    version, _ = store.publish("acme-1", html="second", record={})

    assert version == 2
    assert store.read_html("acme-1", 1) == "original"


def test_corrupt_index_is_treated_as_absent():
    client = FakeS3()
    store = BriefStore(client, BUCKET, "briefs")
    client.objects[store.index_key("acme-1")] = b"{not json"
    assert store.read_index("acme-1") is None
    assert store.publish("acme-1", html="x", record={})[0] == 1


def test_index_written_last_so_it_never_points_at_a_missing_object():
    client = FakeS3()
    store = BriefStore(client, BUCKET, "briefs")
    store.publish("acme-1", html="x", record={})
    assert client.put_calls[-1] == store.index_key("acme-1")


def test_non_404_errors_propagate_from_the_store():
    """Only "missing" is swallowed here; real failures must reach publish_brief."""
    client = FakeS3()
    client.fail_get_with = NotFound("AccessDenied")
    store = BriefStore(client, BUCKET, "briefs")
    with pytest.raises(NotFound):
        store.read_index("acme-1")


# --- publish_brief contract ----------------------------------------------


def test_publish_brief_returns_none_when_disabled():
    settings = make_settings(BRIEFS_ENABLED=False)
    assert publish_brief(make_lead(), make_classification(), settings=settings) is None


def test_publish_brief_returns_none_when_bucket_is_unconfigured():
    settings = make_settings(BRIEFS_S3_BUCKET=None)
    # No client is passed and none can be built, so nothing touches AWS.
    assert publish_brief(make_lead(), make_classification(), settings=settings) is None


def test_publish_brief_happy_path():
    client = FakeS3()
    settings = make_settings()
    lead, classification = make_lead(), make_classification()

    ref = publish_brief(lead, classification, settings=settings, client=client)

    assert isinstance(ref, BriefRef)
    assert ref.version == 1
    assert ref.lead_id == lead_id_for(lead)
    assert ref.s3_key == f"briefs/{ref.lead_id}/v0001.html"
    assert ref.url == f"http://100.79.160.6:8080/briefs/{ref.lead_id}"
    # The URL is version-agnostic on purpose: a re-analysis must not leave a
    # stale link in Slack.
    assert "/v1" not in ref.url

    stored = client.objects[ref.s3_key].decode("utf-8")
    assert "Northgate Manufacturing" in stored
    assert "Version 1" in stored


def test_publish_brief_versions_a_reanalysed_lead():
    client = FakeS3()
    settings = make_settings()
    lead = make_lead()

    first = publish_brief(lead, make_classification(), settings=settings, client=client)
    second = publish_brief(
        lead,
        make_classification(icp_assessment=make_assessment(revenue_band=CriterionStatus.unknown)),
        settings=settings,
        client=client,
    )

    assert (first.version, second.version) == (1, 2)
    assert first.lead_id == second.lead_id
    assert first.url == second.url  # same stable link
    assert first.s3_key != second.s3_key


def test_published_html_footer_matches_the_allocated_version():
    client = FakeS3()
    settings = make_settings()
    lead = make_lead()
    publish_brief(lead, make_classification(), settings=settings, client=client)
    ref = publish_brief(lead, make_classification(), settings=settings, client=client)

    assert "Version 2" in client.objects[ref.s3_key].decode("utf-8")


def test_publish_brief_swallows_storage_failure():
    """Storage must never be the reason a classified lead goes unreported."""
    client = FakeS3()
    client.fail_put_with = RuntimeError("bucket is on fire")
    settings = make_settings()

    assert publish_brief(make_lead(), make_classification(), settings=settings, client=client) is None


def test_publish_brief_swallows_credential_failure():
    client = FakeS3()
    client.fail_get_with = NotFound("ExpiredToken")
    settings = make_settings()

    assert publish_brief(make_lead(), make_classification(), settings=settings, client=client) is None


def test_publish_brief_never_raises_on_unexpected_input():
    """The renderer is total: an object with none of the expected attributes
    still produces a page rather than an exception escaping into the handler."""
    settings = make_settings()
    assert publish_brief(object(), object(), settings=settings, client=FakeS3()) is not None


def test_relative_url_when_no_base_url_is_configured():
    settings = make_settings(BRIEFS_BASE_URL=None, BRIEFS_HTTP_HOST="0.0.0.0")
    ref = publish_brief(make_lead(), make_classification(), settings=settings, client=FakeS3())
    assert ref is not None
    assert ref.url == f"/briefs/{ref.lead_id}"


def test_base_url_derived_from_a_concrete_bind_host():
    settings = make_settings(BRIEFS_BASE_URL=None, BRIEFS_HTTP_HOST="100.79.160.6", BRIEFS_HTTP_PORT=9000)
    assert settings.briefs_effective_base_url() == "http://100.79.160.6:9000"


def test_wildcard_bind_host_yields_no_base_url():
    settings = make_settings(BRIEFS_BASE_URL=None, BRIEFS_HTTP_HOST="0.0.0.0")
    assert settings.briefs_effective_base_url() is None


def test_explicit_base_url_wins_and_trailing_slash_is_normalised():
    settings = make_settings(BRIEFS_BASE_URL="http://briefs.example/", BRIEFS_HTTP_HOST="10.0.0.1")
    assert settings.briefs_effective_base_url() == "http://briefs.example"


# --- Serving -------------------------------------------------------------


def _seeded_store() -> tuple[BriefStore, str]:
    client = FakeS3()
    settings = make_settings()
    lead = make_lead()
    publish_brief(lead, make_classification(), settings=settings, client=client)
    publish_brief(lead, make_classification(), settings=settings, client=client)
    return BriefStore(client, BUCKET, "briefs"), lead_id_for(lead)


def test_health_endpoint():
    store, _ = _seeded_store()
    response = handle_route(store, "/healthz")
    assert response.status == 200
    assert response.body == b"ok\n"


def test_current_route_serves_the_newest_version():
    store, lead_id = _seeded_store()
    response = handle_route(store, brief_path(lead_id))
    assert response.status == 200
    assert response.content_type.startswith("text/html")
    assert "Version 2" in response.body.decode("utf-8")


def test_version_route_serves_an_earlier_brief():
    store, lead_id = _seeded_store()
    response = handle_route(store, brief_path(lead_id, 1))
    assert response.status == 200
    assert "Version 1" in response.body.decode("utf-8")


def test_history_route_lists_every_version_with_links():
    store, lead_id = _seeded_store()
    response = handle_route(store, history_path(lead_id))
    body = response.body.decode("utf-8")

    assert response.status == 200
    assert brief_path(lead_id, 1) in body
    assert brief_path(lead_id, 2) in body
    assert "(current)" in body


def test_json_routes_serve_the_structured_record():
    store, lead_id = _seeded_store()

    current = handle_route(store, f"/briefs/{lead_id}.json")
    assert current.status == 200
    assert json.loads(current.body)["version"] == 2

    pinned = handle_route(store, f"/briefs/{lead_id}/v1.json")
    assert json.loads(pinned.body)["version"] == 1
    assert json.loads(pinned.body)["classification"]["icp_verdict"] == "in_icp"


@pytest.mark.parametrize(
    "path",
    ["/nope", "/briefs/unknown-0000000000", "/briefs/unknown-0000000000/history"],
)
def test_missing_things_are_404(path):
    store, _ = _seeded_store()
    assert handle_route(store, path).status == 404


def test_missing_version_of_a_known_lead_is_404():
    store, lead_id = _seeded_store()
    assert handle_route(store, brief_path(lead_id, 99)).status == 404


# --- Rendering -----------------------------------------------------------


def test_every_criterion_from_icp_fit_is_rendered():
    """
    The page is driven by `icp_fit.CRITERIA`, so a new criterion cannot be
    silently dropped from the brief, and there is no second list of labels
    here to fall out of step with it.
    """
    html = render_brief_html(make_lead(), make_classification(), version=1)
    for spec in CRITERIA:
        assert html_escape(spec.label) in html, spec.label
    assert html.count('class="criterion"') == len(CRITERIA)
    assert html.count('class="gate"') == sum(1 for c in CRITERIA if c.is_gate)


def test_brief_contains_the_whole_analysis():
    classification = make_classification(
        icp_assessment=make_assessment(revenue_band=CriterionStatus.unknown),
    )
    html = render_brief_html(make_lead(), classification, version=4, history_url="/h")

    for expected in (
        "IN ICP" if classification.icp_verdict == ICPVerdict.in_icp else "NEEDS VERIFICATION",
        classification.brief.icp_statement,
        classification.brief.analyst_take,
        classification.brief.opportunity,
        "Procurement cycle may be slow",
        "Modernization &amp; Migration, $250-400k first phase.",
        "Migration Accelerator",
        "They named Snowflake themselves",
        "Tier-two automotive supplier.",
        "VP, Data &amp; Analytics",
        "Credible mid-market buyer",
        "We need help migrating our warehouse to Snowflake.",
        "Why in ICP",
        "Unverified",
        "Version 4",
        'href="/h"',
    ):
        assert expected in html, expected


def test_brief_is_self_contained():
    html = render_brief_html(make_lead(), make_classification(), version=1)
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "<script" not in html
    assert "http://" not in html.replace("http://100.79", "")  # no CDN or asset hosts
    assert 'name="viewport"' in html
    assert "noindex" in html


def test_untrusted_lead_content_is_escaped():
    lead = make_lead(
        company='<script>alert("x")</script>',
        message="<img src=x onerror=alert(1)>",
    )
    html = render_brief_html(lead, make_classification(company=None), version=1)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    # The payload survives as text but never as markup.
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_brief_renders_without_an_icp_assessment():
    """Spam never gets an assessment; the page must still render."""
    minimal = apply_icp_fit(
        EnrichedLeadClassification(label=LeadLabel.ignore, reason="Vendor pitch.")
    )
    html = render_brief_html(HubSpotLead(raw_text="buy my seo"), minimal, version=1)
    assert "NOT EVALUATED" in html
    assert "ICP criteria" not in html


def test_history_page_handles_an_empty_lead():
    html = render_history_html("acme-1234567890", [])
    assert "No briefs have been published" in html
