"""
Tests for the two Slack renderers in `core.processor`.

These pin the reason the split exists: the card must stay short enough that
Slack never collapses it, and the detailed brief must always be postable
despite Slack's 3,000-character block limit. All offline — no LLM, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from leads_agent.core.processor import (
    CARD_MAX_CRITERIA,
    SLACK_BLOCK_LIMIT,
    SLACK_CHUNK_LIMIT,
    ProcessedLead,
    chunk_for_slack,
    format_full_brief,
    format_slack_message,
    has_full_brief,
    post_to_slack,
)
from leads_agent.icp_fit import CRITERIA, apply_icp_fit
from leads_agent.models import (
    CompanyResearch,
    CriterionStatus,
    EnrichedLeadClassification,
    HubSpotLead,
    ICPAssessment,
    ICPCriterion,
    LeadClassification,
    LeadLabel,
    OutreachBrief,
)

ALL_FIELDS = [c.field for c in CRITERIA]
HARD_GATES = [c.field for c in CRITERIA if c.is_gate and c.hard]

# Deliberately verbose, in the shape real model output takes — the card's job
# is to survive text like this, so the fixtures must not be tidier than
# production.
LONG_FINDING = (
    "Searches for the company name and for the contact returned no usable results — no "
    "website, no press coverage, no filings, no directory listing and no job postings, "
    "which is itself a red flag for a business claiming this scale."
)
LONG_STATEMENT = (
    "Pre-revenue, self-funded solo-founder sports-AI startup with revenue well below the "
    "$250M floor, no company web presence found, a no-code MVP on Bubble/Make.com/Airtable "
    "plus the Gemini API, no data platform and no executive sponsor beyond the founder, "
    "closest to a Predictive Automation build but entering as a paid MVP audit of unstated size."
)
LONG_TAKE = (
    "This is a clean, articulate inbound and the founder has done more real work than most "
    "who write in. It is also, plainly, out of ICP: pre-revenue, gmail address, no findable "
    "company, a no-code MVP and a self-funded budget he cannot quantify. Our floor exists "
    "precisely because engagements like this cannot support a $150k+ build. Recommend a "
    "courteous decline with a referral to a boutique shop."
)

LEAD = HubSpotLead(
    first_name="Luke",
    last_name="Koman",
    email="luke@example.com",
    company="InReach AI",
    message="We have a working MVP and want a partner to audit and professionalize it. " * 6,
)


def _criterion(status: CriterionStatus) -> ICPCriterion:
    return ICPCriterion(
        status=status,
        finding=LONG_FINDING,
        evidence=None if status == CriterionStatus.unknown else "Lead's own words plus research.",
    )


def build_assessment(default: CriterionStatus = CriterionStatus.met, **overrides) -> ICPAssessment:
    fields = {name: _criterion(default) for name in ALL_FIELDS}
    for name, status in overrides.items():
        fields[name] = _criterion(status)
    return ICPAssessment(**fields)


def build_brief() -> OutreachBrief:
    return OutreachBrief(
        icp_statement=LONG_STATEMENT,
        analyst_take=LONG_TAKE,
        opportunity="A referenceable proof point in a focus overlay if it ever scales.",
        risks=[LONG_FINDING] * 4,
        recommended_entry="Modernization & Migration, foundation build, $150k-$300k band.",
        talking_points=[LONG_FINDING] * 4,
        accelerators=["Snowflake foundation blueprint", "Agent evaluation harness"],
    )


def build_classification(
    *,
    assessment: ICPAssessment | None = None,
    brief: OutreachBrief | None = None,
    label: LeadLabel = LeadLabel.promising,
    research: bool = True,
) -> EnrichedLeadClassification:
    """An enriched classification with the derived ICP fields already applied."""
    classification = EnrichedLeadClassification(
        label=label,
        reason="Genuine inbound inquiry about a computer-vision build; no disqualifier triggered.",
        lead_summary="Founder of an early-stage AI startup seeking a build partner.",
        key_signals=["genuine service inquiry", "pre-revenue startup", "founder is the buyer"],
        company_research=(
            CompanyResearch(
                company_name="InReach AI",
                company_description="Self-described AI-powered film analysis platform. " * 4,
                industry="Sports technology",
                website="inreach.ai",
            )
            if research
            else None
        ),
        research_summary=("Web research returned no results for the company. " * 6) if research else None,
        icp_assessment=assessment,
        brief=brief,
    )
    return apply_icp_fit(classification)


def out_of_icp() -> EnrichedLeadClassification:
    return build_classification(
        assessment=build_assessment(**{gate: CriterionStatus.not_met for gate in HARD_GATES}),
        brief=build_brief(),
    )


def needs_verification() -> EnrichedLeadClassification:
    return build_classification(
        assessment=build_assessment(**{gate: CriterionStatus.unknown for gate in HARD_GATES[:2]}),
        brief=build_brief(),
    )


# --- Card size ------------------------------------------------------------


@pytest.mark.parametrize("factory", [out_of_icp, needs_verification])
def test_card_stays_well_under_slack_collapse_threshold(factory):
    """Slack hides content past ~4,000 chars behind "Show more" — the card must never get there."""
    card = format_slack_message(LEAD, factory())
    assert len(card) < 1200, card
    assert len(card.splitlines()) <= 14, card


def test_card_is_a_fraction_of_the_full_brief():
    classification = out_of_icp()
    card = format_slack_message(LEAD, classification)
    brief = format_full_brief(LEAD, classification)
    assert len(card) * 3 < len(brief)


def test_card_omits_the_detail_that_belongs_in_the_brief():
    classification = out_of_icp()
    card = format_slack_message(LEAD, classification)
    for absent in ("Talking points", "Risks", "Company Research", "Research summary", "Accelerators"):
        assert absent not in card
    # Non-gate criteria never decide a verdict, so they never reach the card.
    assert "Focus overlay" not in card


def test_card_does_not_truncate_mid_word():
    card = format_slack_message(LEAD, out_of_icp())
    for line in card.splitlines():
        if line.endswith("…"):
            # An ellipsis must follow a complete word, not a severed one.
            assert line[-2] not in "-—,;:"


# --- Deciding criteria ----------------------------------------------------


def test_out_of_icp_card_lists_failed_gates_with_overflow():
    card = format_slack_message(LEAD, out_of_icp())
    assert "⛔ *NOT IN ICP*" in card
    assert "*Fails:*" in card
    bullets = [line for line in card.splitlines() if line.startswith("• ")]
    assert len(bullets) == CARD_MAX_CRITERIA
    # Five hard gates fail, so two must be rolled up rather than listed.
    assert f"_+{len(HARD_GATES) - CARD_MAX_CRITERIA} more in the brief_" in card


def test_needs_verification_card_lists_what_is_unverified():
    card = format_slack_message(LEAD, needs_verification())
    assert "🔍 *NEEDS VERIFICATION*" in card
    assert "*Unverified:*" in card
    assert "*Fails:*" not in card
    bullets = [line for line in card.splitlines() if line.startswith("• ")]
    assert len(bullets) == 2
    assert "more in the brief" not in card


def test_in_icp_card_lists_qualifying_gates():
    card = format_slack_message(LEAD, build_classification(assessment=build_assessment(), brief=build_brief()))
    assert "✅ *IN ICP*" in card
    assert "*Qualifies:*" in card
    assert "*Action:* `follow_up`" in card


def test_card_shows_no_criteria_block_without_an_assessment():
    card = format_slack_message(LEAD, build_classification(brief=build_brief(), research=False))
    assert "*Fails:*" not in card
    assert "*Qualifies:*" not in card


# --- Triage-only and spam paths -------------------------------------------


def test_spam_card_is_the_triage_verdict_only():
    """No assessment means `not_evaluated`; the card must not show that as an ICP verdict."""
    classification = build_classification(label=LeadLabel.ignore, research=False)
    card = format_slack_message(LEAD, classification)
    assert "🚫 *IGNORE* — not a genuine inquiry" in card
    assert "NOT EVALUATED" not in card
    assert "*Action:* `ignore`" in card
    assert len(card) < 500
    assert not has_full_brief(classification)


def test_triage_only_promising_card_states_go():
    """A plain LeadClassification (research skipped) has no verdict to defer to."""
    classification = LeadClassification(label=LeadLabel.promising, reason="Genuine inbound inquiry.")
    card = format_slack_message(LEAD, classification)
    assert "✅ *GO* — genuine inquiry" in card
    assert not has_full_brief(classification)


def test_card_does_not_restate_go_when_a_verdict_follows():
    card = format_slack_message(LEAD, out_of_icp())
    assert "*GO*" not in card


def test_card_renders_without_a_brief():
    classification = build_classification(assessment=build_assessment(**{HARD_GATES[0]: CriterionStatus.not_met}))
    card = format_slack_message(LEAD, classification)
    assert "⛔ *NOT IN ICP*" in card
    assert "*Fails:*" in card
    assert "🧠" not in card


def test_full_brief_renders_without_a_brief_or_assessment():
    """
    Renders without an assessment. It no longer repeats the card's triage line
    — that put "GO" directly under "NOT IN ICP" — so it carries only detail.
    """
    classification = build_classification(label=LeadLabel.ignore, research=False)
    brief = format_full_brief(LEAD, classification)
    assert classification.lead_summary in brief
    assert "GO" not in brief and "IGNORE" not in brief


# --- brief_url ------------------------------------------------------------


def test_brief_url_renders_a_link_line():
    card = format_slack_message(LEAD, out_of_icp(), brief_url="https://briefs.example.com/b/abc")
    assert "<https://briefs.example.com/b/abc|Full brief →>" in card
    assert card.splitlines()[-1].startswith("*Action:*")


def test_brief_url_is_optional():
    assert "Full brief" not in format_slack_message(LEAD, out_of_icp())


def test_brief_url_renders_without_an_action():
    classification = LeadClassification(label=LeadLabel.promising, reason="Genuine inbound inquiry.")
    card = format_slack_message(LEAD, classification, brief_url="https://briefs.example.com/b/abc")
    assert card.splitlines()[-1] == "<https://briefs.example.com/b/abc|Full brief →>"


def test_lead_info_header_is_one_line():
    card = format_slack_message(LEAD, out_of_icp(), include_lead_info=True)
    first = card.splitlines()[0]
    assert "Luke Koman" in first
    assert "InReach AI" in first
    assert len(card) < 1400


# --- Chunking -------------------------------------------------------------


def test_chunks_respect_the_block_limit_and_lose_nothing():
    brief = format_full_brief(LEAD, out_of_icp(), include_lead_info=True)
    chunks = chunk_for_slack(brief)
    assert chunks
    assert all(len(chunk) <= SLACK_CHUNK_LIMIT < SLACK_BLOCK_LIMIT for chunk in chunks)
    assert "\n".join(chunks) == brief


def test_chunking_splits_on_line_boundaries():
    text = "\n".join(f"line {i} " + "x" * 90 for i in range(80))
    chunks = chunk_for_slack(text, limit=500)
    assert len(chunks) > 1
    assert all(chunk.startswith("line ") for chunk in chunks)


def test_an_over_long_single_line_is_split_rather_than_dropped():
    text = "word " * 400
    chunks = chunk_for_slack(text.strip(), limit=200)
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "\n".join(chunks).split() == text.split()


def test_empty_text_produces_no_chunks():
    assert chunk_for_slack("") == []
    assert chunk_for_slack("   \n  ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_for_slack("hello") == ["hello"]


# --- Posting --------------------------------------------------------------


@dataclass
class FakeSettings:
    dry_run: bool = False


@dataclass
class FakeSlackClient:
    """Records posts instead of making them. Never touches the network."""

    posts: list[dict[str, Any]] = field(default_factory=list)
    reactions: list[dict[str, Any]] = field(default_factory=list)

    # Slack SDK method name, deliberately camelCase.
    def chat_postMessage(self, **kwargs) -> dict[str, Any]:
        self.posts.append(kwargs)
        return {"ok": True, "ts": f"1700000000.{len(self.posts):06d}"}

    def reactions_add(self, **kwargs) -> dict[str, Any]:
        self.reactions.append(kwargs)
        return {"ok": True}


@pytest.fixture
def fake_client(monkeypatch) -> FakeSlackClient:
    client = FakeSlackClient()
    monkeypatch.setattr("leads_agent.core.processor.slack_client", lambda settings: client)
    return client


def _processed(classification: LeadClassification | EnrichedLeadClassification) -> ProcessedLead:
    return ProcessedLead(
        lead=LEAD,
        classification=classification,
        slack_message=format_slack_message(LEAD, classification),
        full_brief=format_full_brief(LEAD, classification),
    )


def test_post_to_slack_posts_card_then_brief_in_thread(fake_client):
    processed = _processed(out_of_icp())
    post_to_slack(FakeSettings(), processed, channel_id="C1", thread_ts="1000.0001")

    card, *replies = fake_client.posts
    assert card["text"] == processed.slack_message
    assert card["thread_ts"] == "1000.0001"
    assert replies, "the full brief must be posted beneath the card"
    # The card is itself a reply to the HubSpot message, so the brief joins that thread.
    assert all(reply["thread_ts"] == "1000.0001" for reply in replies)
    assert "\n".join(reply["text"] for reply in replies) == processed.full_brief


def test_post_to_slack_threads_the_brief_under_the_card_when_top_level(fake_client):
    post_to_slack(FakeSettings(), _processed(out_of_icp()), channel_id="C1")

    card, *replies = fake_client.posts
    assert "thread_ts" not in card
    assert replies and all(reply["thread_ts"] == "1700000000.000001" for reply in replies)


def test_post_to_slack_skips_the_thread_reply_for_spam(fake_client):
    classification = build_classification(label=LeadLabel.ignore, research=False)
    post_to_slack(FakeSettings(), _processed(classification), channel_id="C1", thread_ts="1000.0001")
    assert len(fake_client.posts) == 1


def test_post_to_slack_renders_the_brief_url_on_the_card(fake_client):
    post_to_slack(
        FakeSettings(),
        _processed(out_of_icp()),
        channel_id="C1",
        brief_url="https://briefs.example.com/b/abc",
    )
    assert "Full brief →" in fake_client.posts[0]["text"]


def test_dry_run_posts_nothing(fake_client):
    post_to_slack(FakeSettings(dry_run=True), _processed(out_of_icp()), channel_id="C1", thread_ts="1000.0001")
    assert fake_client.posts == []
