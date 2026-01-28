import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import logfire
from opentelemetry import trace

from leads_agent.agent import classify_lead
from leads_agent.models import EnrichedLeadClassification, HubSpotLead, LeadClassification
from leads_agent.slack import slack_client

if TYPE_CHECKING:
    from leads_agent.config import Settings

# Configure logfire only if token is available
_logfire_enabled = bool(os.environ.get("LOGFIRE_TOKEN"))
if _logfire_enabled:
    try:
        logfire.configure()
    except Exception:
        # If configuration fails, disable logfire
        _logfire_enabled = False


@contextmanager
def _logfire_span(name: str, **kwargs):
    """Context manager for logfire spans that works even when logfire is disabled."""
    if _logfire_enabled:
        with logfire.span(name, **kwargs):
            yield
    else:
        yield


@dataclass
class ProcessedLead:
    """Result of processing a lead."""

    lead: HubSpotLead
    classification: LeadClassification | EnrichedLeadClassification
    slack_message: str

    @property
    def label(self) -> str:
        return self.classification.label.value

    @property
    def is_promising(self) -> bool:
        return self.label == "promising"


def format_slack_message(
    lead: HubSpotLead,
    classification: LeadClassification | EnrichedLeadClassification,
    include_lead_info: bool = False,
) -> str:
    """
    Format classification result as a Slack message.

    Args:
        lead: The parsed lead data
        classification: The classification result
        include_lead_info: If True, include lead details (for test channel posts)
    """
    parts = []

    # Optionally include lead info header (for test mode)
    if include_lead_info:
        name = f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "Unknown"
        email = lead.email
        email_display = f"<mailto:{email}|{email}>" if email else "no email"
        parts.append(f"*Lead:* {name} ({email_display})")
        if lead.company:
            parts.append(f"*Company:* {lead.company}")
        if lead.message:
            msg_preview = lead.message[:150] + "..." if len(lead.message) > 150 else lead.message
            parts.append(f"*Message:* {msg_preview}")
        parts.append("")  # blank line

    # Go / No-go (hide taxonomy)
    if classification.label.value == "promising":
        parts.append(f"✅ *GO* ({classification.confidence:.0%})")
    else:
        parts.append(f"🚫 *IGNORE* ({classification.confidence:.0%})")
    parts.append(f"_{classification.reason}_")

    # Optional final score (for promising leads after research+scoring)
    if getattr(classification, "score", None) is not None and getattr(classification, "action", None) is not None:
        parts.append(f"\n⭐ *Score:* {classification.score}/5 · *Action:* {classification.action.value}")
        if getattr(classification, "score_reason", None):
            parts.append(f"_{classification.score_reason}_")

    # Optional lead summary/signals (useful when triage output includes them)
    if classification.lead_summary:
        parts.append(f"\n*🧾 Summary:* {classification.lead_summary}")
    if classification.key_signals:
        parts.append("\n*🏷️ Signals:* " + ", ".join(classification.key_signals))

    # Extracted company if different
    if classification.company and classification.company != lead.company:
        parts.append(f"\n📋 Company: {classification.company}")

    # Enrichment results
    if isinstance(classification, EnrichedLeadClassification):
        if classification.company_research:
            cr = classification.company_research
            parts.append("\n*📊 Company Research:*")
            parts.append(f"• *{cr.company_name}*: {cr.company_description}")
            if cr.industry:
                parts.append(f"• Industry: {cr.industry}")
            if cr.company_size:
                parts.append(f"• Size: {cr.company_size}")
            if cr.website:
                # Format URL for Slack clickability
                url = cr.website if cr.website.startswith("http") else f"https://{cr.website}"
                parts.append(f"• Website: <{url}|{cr.website}>")
            if cr.relevance_notes:
                parts.append(f"• Relevance: {cr.relevance_notes}")

        if classification.contact_research:
            cr = classification.contact_research
            parts.append("\n*👤 Contact Research:*")
            title_str = f" - {cr.title}" if cr.title else ""
            parts.append(f"• *{cr.full_name}*{title_str}")
            if cr.linkedin_summary:
                summary = cr.linkedin_summary[:300] + "..." if len(cr.linkedin_summary) > 300 else cr.linkedin_summary
                parts.append(f"• {summary}")
            if cr.relevance_notes:
                parts.append(f"• Relevance: {cr.relevance_notes}")

        if classification.research_summary:
            parts.append(f"\n*📝 Summary:*\n{classification.research_summary}")

    return "\n".join(parts)


def process_lead(
    settings: "Settings",
    lead: HubSpotLead,
    *,
    max_searches: int = 4,
) -> ProcessedLead:
    """
    Process a single lead: classify and format response.

    Args:
        settings: Application settings
        lead: Parsed HubSpot lead
        max_searches: Max web searches for enrichment

    Returns:
        ProcessedLead with classification and formatted Slack message
    """
    classification = classify_lead(settings, lead, max_searches=max_searches)

    # Handle ClassificationResult wrapper (from debug mode)
    if hasattr(classification, "classification"):
        classification = classification.classification

    slack_message = format_slack_message(lead, classification, include_lead_info=False)

    return ProcessedLead(
        lead=lead,
        classification=classification,
        slack_message=slack_message,
    )


def post_to_slack(
    settings: "Settings",
    processed: ProcessedLead,
    *,
    channel_id: str,
    thread_ts: str | None = None,
    include_lead_info: bool = False,
) -> None:
    """
    Post processed lead result to Slack.

    Args:
        settings: Application settings
        processed: The processed lead result
        channel_id: Slack channel ID to post to
        thread_ts: If provided, post as thread reply; otherwise post to main channel
        include_lead_info: If True, include lead details in message
    """
    if settings.dry_run:
        print(f"[DRY RUN] Would post to {channel_id}" + (f" (thread: {thread_ts})" if thread_ts else ""))
        return

    # Re-format with lead info if needed
    message = (
        format_slack_message(processed.lead, processed.classification, include_lead_info=include_lead_info)
        if include_lead_info
        else processed.slack_message
    )

    client = slack_client(settings)

    kwargs = {
        "channel": channel_id,
        "text": message,
    }
    if thread_ts:
        kwargs["thread_ts"] = thread_ts

    client.chat_postMessage(**kwargs)


def process_and_post(
    settings: "Settings",
    lead: HubSpotLead,
    *,
    channel_id: str,
    thread_ts: str | None = None,
    max_searches: int = 4,
    include_lead_info: bool = False,
) -> ProcessedLead:
    """
    Process a lead and post the result to Slack.

    This is the main entry point for both production and testing modes.

    Args:
        settings: Application settings
        lead: Parsed HubSpot lead
        channel_id: Where to post the result
        thread_ts: If provided, post as thread reply (production mode)
        max_searches: Max web searches for enrichment
        include_lead_info: Include lead details in message (test mode)

    Returns:
        ProcessedLead with results
    """
    # Group all agent traces (triage/research/scoring) and Slack posting under one lead span.
    email_domain = ""
    if lead.email and "@" in lead.email:
        email_domain = lead.email.split("@", 1)[1].lower()

    # Prefer Slack timestamp when available; otherwise fall back to a stable short hash.
    trace_id = thread_ts or (lead.email.lower() if lead.email else "")
    if not trace_id:
        base = "|".join(
            [
                lead.company or "",
                email_domain,
                lead.first_name or "",
                lead.last_name or "",
                (lead.message or lead.raw_text or "")[:500],
            ]
        )
        trace_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

    current = trace.get_current_span()
    has_parent = current.get_span_context().is_valid

    # Only create a top-level lead.process span if we aren't already inside one.
    span_name = "lead.post" if has_parent else "lead.process"

    with _logfire_span(
        span_name,
        lead_id=trace_id,
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        email=lead.email,
        email_domain=email_domain,
        company=lead.company,
        max_searches=max_searches,
        include_lead_info=include_lead_info,
        dry_run=settings.dry_run,
    ):
        processed = process_lead(settings, lead, max_searches=max_searches)

        post_to_slack(
            settings,
            processed,
            channel_id=channel_id,
            thread_ts=thread_ts,
            include_lead_info=include_lead_info,
        )

        return processed
