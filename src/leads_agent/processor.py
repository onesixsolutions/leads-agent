"""
Lead processing pipeline — shared between production and testing modes.

Production: Event from Slack → process → post as thread reply
Testing: Pull history → process → post to test channel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .llm import classify_lead
from .models import EnrichedLeadClassification, HubSpotLead, LeadClassification
from .slack import slack_client

if TYPE_CHECKING:
    from .config import Settings


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
    label_emoji = {
        "spam": "🔴",
        "solicitation": "🟡",
        "promising": "🟢",
    }.get(classification.label.value, "⚪")

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

    # Classification result
    parts.append(f"{label_emoji} *{classification.label.value.upper()}* ({classification.confidence:.0%})")
    parts.append(f"_{classification.reason}_")

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
    enrich: bool = False,
    max_searches: int = 4,
) -> ProcessedLead:
    """
    Process a single lead: classify and format response.

    Args:
        settings: Application settings
        lead: Parsed HubSpot lead
        enrich: Whether to research promising leads
        max_searches: Max web searches for enrichment

    Returns:
        ProcessedLead with classification and formatted Slack message
    """
    classification = classify_lead(settings, lead, enrich=enrich, max_searches=max_searches)

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
    enrich: bool = False,
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
        enrich: Whether to research promising leads
        max_searches: Max web searches for enrichment
        include_lead_info: Include lead details in message (test mode)

    Returns:
        ProcessedLead with results
    """
    processed = process_lead(settings, lead, enrich=enrich, max_searches=max_searches)

    post_to_slack(
        settings,
        processed,
        channel_id=channel_id,
        thread_ts=thread_ts,
        include_lead_info=include_lead_info,
    )

    return processed
