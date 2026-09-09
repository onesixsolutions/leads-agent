"""
Lead briefs: render the full analysis as HTML, keep every version in S3, serve
it back over HTTP.

The Slack card is the decision; the brief is the evidence. `publish_brief` is
the whole integration surface — call it after classification, get a URL to put
in the Slack message, and carry on regardless of what S3 did.

Storage is deliberately best-effort. A lead that was successfully analysed must
still reach a human even if the bucket is misconfigured, the IAM role has
expired or boto3 is not installed, so every failure path here ends in `None`
and a log line, never an exception.

Module layout
-------------
`identity`  stable lead id (what makes two runs the same lead)
`render`    HTML generation (no I/O)
`storage`   the S3 key scheme and version allocation (no rendering)
`routes`    the URL scheme, as pure functions shared by publisher and server
`server`    a stdlib HTTP listener that maps routes onto the store
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from leads_agent.briefs.identity import is_valid_lead_id, lead_id_for
from leads_agent.briefs.render import render_brief_html, render_history_html
from leads_agent.briefs.routes import (
    Route,
    RouteKind,
    absolute_url,
    brief_path,
    history_path,
    resolve_route,
)
from leads_agent.briefs.storage import BriefIndex, BriefStore, BriefVersion

if TYPE_CHECKING:
    from leads_agent.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "BriefIndex",
    "BriefRef",
    "BriefStore",
    "BriefVersion",
    "Route",
    "RouteKind",
    "absolute_url",
    "brief_path",
    "brief_store",
    "history_path",
    "is_valid_lead_id",
    "lead_id_for",
    "publish_brief",
    "render_brief_html",
    "render_history_html",
    "resolve_route",
]


@dataclass(frozen=True)
class BriefRef:
    """Where a published brief lives, and how to link to it."""

    lead_id: str
    version: int
    url: str
    s3_key: str


def _s3_client(settings: Settings) -> Any | None:
    """
    Build an S3 client from the default boto3 credential chain.

    Credentials are never read from our own settings on purpose: on the EC2
    host they come from the instance role, and locally from a profile or the
    environment. Returns None — rather than raising — when boto3 is absent or
    no credentials can be resolved.
    """
    try:
        import boto3
    except ImportError:
        logger.warning("Briefs enabled but boto3 is not installed; skipping publish")
        return None

    kwargs: dict[str, Any] = {}
    region = settings.briefs_s3_region
    if region:
        kwargs["region_name"] = region
        # Presigning against the global endpoint makes S3 answer 307 and
        # redirect to the regional host, which invalidates the signature.
        # Pinning the regional endpoint and SigV4 keeps signed links working.
        kwargs["endpoint_url"] = f"https://s3.{region}.amazonaws.com"
    try:
        from botocore.client import Config

        kwargs["config"] = Config(signature_version="s3v4")
    except ImportError:  # pragma: no cover - botocore ships with boto3
        pass
    return boto3.client("s3", **kwargs)


def _brief_link(
    settings: Settings,
    store: BriefStore,
    lead_id: str,
    s3_key: str,
    base_url: str | None,
) -> str:
    """
    The URL that goes on the Slack card.

    `presigned` works from anywhere with nothing running, at the cost of
    expiring. `app` is stable but only resolves where the listener is
    reachable. Falls back to the app path if signing fails, so a link problem
    never costs us the brief.
    """
    if settings.briefs_link_mode == "presigned":
        try:
            return store.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": store.bucket,
                    "Key": s3_key,
                    # Without this S3 serves the object as a download rather
                    # than rendering it in the browser.
                    "ResponseContentType": "text/html; charset=utf-8",
                },
                ExpiresIn=settings.briefs_presigned_ttl_s,
            )
        except Exception:
            logger.exception("Could not presign the brief; falling back to the app URL")

    path = brief_path(lead_id)
    if not base_url:
        logger.warning(
            "BRIEFS_BASE_URL is not set; brief link is relative (%s) and will "
            "not be clickable from Slack",
            path,
        )
        return path
    return absolute_url(base_url, path)


def brief_store(settings: Settings, client: Any | None = None) -> BriefStore | None:
    """
    A `BriefStore` for the configured bucket, or None when briefs are off.

    Args:
        settings: Application settings.
        client: Pre-built S3 client. Supplied by the tests and by the HTTP
            server, which builds one client for its lifetime rather than one
            per request.
    """
    if not settings.briefs_enabled:
        return None
    if not settings.briefs_s3_bucket:
        logger.warning("BRIEFS_ENABLED is set but BRIEFS_S3_BUCKET is empty")
        return None

    client = client or _s3_client(settings)
    if client is None:
        return None

    return BriefStore(client, settings.briefs_s3_bucket, settings.briefs_s3_prefix)


def _structured_record(
    lead: Any,
    classification: Any,
    *,
    lead_id: str,
    version: int,
    generated_at: datetime,
) -> dict[str, Any]:
    """
    The machine-readable twin of the HTML.

    The prose answers "what about this lead?"; this answers "what about all our
    leads?" — which ICP gate fails most often, how often a verdict changes
    between versions — without anyone having to parse a rendered page.
    """

    def dump(model: Any) -> Any:
        if hasattr(model, "model_dump"):
            return model.model_dump(mode="json")
        return model

    return {
        "lead_id": lead_id,
        "version": version,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "lead": dump(lead),
        "classification": dump(classification),
    }


def publish_brief(
    lead: Any,
    classification: Any,
    *,
    settings: Settings,
    client: Any | None = None,
) -> BriefRef | None:
    """
    Render, store and link a new version of this lead's brief.

    Never raises: a storage problem returns None and the caller posts to Slack
    without a link. Losing the brief is an inconvenience; losing the lead is
    not acceptable.

    Args:
        lead: The parsed `HubSpotLead`.
        classification: The (usually enriched) classification to render.
        settings: Application settings.
        client: Pre-built S3 client, for tests.

    Returns:
        A `BriefRef` with the version-stable S3 key and the URL to share, or
        None when briefs are disabled, unconfigured, or storage failed.
    """
    try:
        store = brief_store(settings, client)
        if store is None:
            return None

        lead_id = lead_id_for(lead)
        generated_at = datetime.now(UTC)
        base_url = settings.briefs_effective_base_url()

        # The version number is only known after allocation, so render against
        # the number the store hands back rather than guessing it. One extra
        # read of the index is cheaper than a page whose footer lies.
        index = store.read_index(lead_id)
        version = store.next_version(lead_id, index)

        html = render_brief_html(
            lead,
            classification,
            version=version,
            generated_at=generated_at,
            history_url=(
                absolute_url(base_url, history_path(lead_id))
                if base_url
                else history_path(lead_id)
            ),
        )
        record = _structured_record(
            lead,
            classification,
            lead_id=lead_id,
            version=version,
            generated_at=generated_at,
        )

        verdict = getattr(classification, "icp_verdict", None)
        action = getattr(classification, "action", None)
        company_research = getattr(classification, "company_research", None)

        published_version, s3_key = store.publish(
            lead_id,
            html=html,
            record=record,
            verdict=getattr(verdict, "value", None),
            action=getattr(action, "value", None),
            company=(
                getattr(company_research, "company_name", None)
                or getattr(classification, "company", None)
                or getattr(lead, "company", None)
            ),
            contact=f"{getattr(lead, 'first_name', '') or ''} {getattr(lead, 'last_name', '') or ''}".strip()
            or None,
        )

        url = _brief_link(settings, store, lead_id, s3_key, base_url)

        logger.info("Published brief %s v%s -> s3://%s/%s", lead_id, published_version, store.bucket, s3_key)
        return BriefRef(lead_id=lead_id, version=published_version, url=url, s3_key=s3_key)

    except Exception:
        # Deliberately broad: this is the boundary between "nice to have" and
        # "must not break lead processing".
        logger.exception("Failed to publish lead brief; continuing without one")
        return None
