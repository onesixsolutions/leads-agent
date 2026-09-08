import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from slack_sdk.errors import SlackApiError

from leads_agent.agent import classify_lead, enrich_lead, triage_lead
from leads_agent.icp_fit import verdict_display
from leads_agent.models import (
    EnrichedLeadClassification,
    HubSpotLead,
    ICPVerdict,
    LeadClassification,
)
from leads_agent.slack import slack_client

if TYPE_CHECKING:
    from leads_agent.config import Settings

logger = logging.getLogger(__name__)

# Slack emoji names (without surrounding colons) used to signal triage outcome
# on the original HubSpot message.
REACTION_PROMISING = "white_check_mark"
REACTION_IGNORED = "x"


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

    # Stage 1: intent go/no-go. Binary — no confidence percentage.
    if classification.label.value == "promising":
        parts.append("✅ *GO* — genuine inquiry")
    else:
        parts.append("🚫 *IGNORE* — not a genuine inquiry")
    parts.append(f"_{classification.reason}_")

    # Stage 2: ICP verdict, with the criteria that produced it.
    verdict = getattr(classification, "icp_verdict", None)
    if verdict is not None and verdict != ICPVerdict.not_evaluated:
        emoji, label = verdict_display(verdict)
        parts.append(f"\n{emoji} *{label}*")

        brief = getattr(classification, "brief", None)
        if brief and brief.icp_statement:
            parts.append(f"*{brief.icp_statement}*")

        # Why it is / isn't in ICP — the point of the whole exercise.
        reasons_out = getattr(classification, "reasons_out_of_icp", None)
        if reasons_out:
            parts.append("\n*Why not in ICP:*")
            parts.extend(f"• {r}" for r in reasons_out)

        reasons_in = getattr(classification, "reasons_in_icp", None)
        if reasons_in:
            parts.append("\n*Why in ICP:*")
            parts.extend(f"• {r}" for r in reasons_in)

        open_qs = getattr(classification, "open_questions", None)
        if open_qs:
            parts.append("\n*Unverified — find out before pursuing:*")
            parts.extend(f"• {q}" for q in open_qs)

        # Judgement layer.
        if brief:
            if brief.analyst_take:
                parts.append(f"\n*🧠 Take:* {brief.analyst_take}")
            if brief.opportunity:
                parts.append(f"*📈 Opportunity:* {brief.opportunity}")
            if brief.risks:
                parts.append("*⚠️ Risks:* " + "; ".join(brief.risks))
            if brief.exception_case:
                parts.append(f"*🚩 Exception case:* {brief.exception_case}")
            if brief.recommended_entry:
                parts.append(f"\n*🚪 Recommended entry:* {brief.recommended_entry}")
            if brief.accelerators:
                parts.append("*🧰 Accelerators:* " + ", ".join(brief.accelerators))
            if brief.talking_points:
                parts.append("\n*💬 Talking points:*")
                parts.extend(f"• {t}" for t in brief.talking_points)

        action = getattr(classification, "action", None)
        if action is not None:
            parts.append(f"\n*Action:* `{action.value}`")

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
            parts.append(f"\n*📝 Research summary:*\n{classification.research_summary}")

    return "\n".join(parts)


def process_lead(
    settings: "Settings",
    lead: HubSpotLead,
    *,
    max_searches: int = 4,
    skip_research: bool = False,
) -> ProcessedLead:
    """
    Process a single lead: classify and format response.

    Args:
        settings: Application settings
        lead: Parsed HubSpot lead
        max_searches: Max web searches for enrichment
        skip_research: If True, run triage only (no web research or assessment)

    Returns:
        ProcessedLead with classification and formatted Slack message
    """
    if skip_research:
        classification: LeadClassification | EnrichedLeadClassification = triage_lead(settings, lead)
    else:
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
        format_slack_message(
            processed.lead,
            processed.classification,
            include_lead_info=include_lead_info,
        )
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


def react_to_lead_message(
    settings: "Settings",
    *,
    channel_id: str,
    timestamp: str,
    is_promising: bool,
    client=None,
) -> None:
    """
    Add an emoji reaction to the original lead message indicating triage outcome.

    Uses ✅ (`white_check_mark`) for promising leads and ❌ (`x`) for ignored
    leads. Failures are logged but never raised — the thread reply is the
    primary signal and the reaction is purely additive.
    """
    emoji = REACTION_PROMISING if is_promising else REACTION_IGNORED

    if settings.dry_run:
        print(f"[DRY RUN] Would react :{emoji}: on {channel_id}/{timestamp}")
        return

    client = client or slack_client(settings)
    try:
        client.reactions_add(channel=channel_id, timestamp=timestamp, name=emoji)
    except SlackApiError as e:
        error = e.response.get("error", "unknown") if e.response else "unknown"
        # `already_reacted` is benign (e.g., reprocessing the same event) — log
        # at debug instead of warning so it doesn't look like a failure.
        if error == "already_reacted":
            logger.debug("Reaction :%s: already present on %s/%s", emoji, channel_id, timestamp)
        else:
            logger.warning("Failed to add :%s: reaction on %s/%s: %s", emoji, channel_id, timestamp, error)


def process_and_post(
    settings: "Settings",
    lead: HubSpotLead,
    *,
    channel_id: str,
    thread_ts: str | None = None,
    max_searches: int = 4,
    include_lead_info: bool = False,
    skip_research: bool = False,
) -> ProcessedLead:
    """
    Process a lead and post the result to Slack.

    Reacts to the original message immediately after triage, before any
    web research runs, so the outcome is visible as early as possible.

    Args:
        settings: Application settings
        lead: Parsed HubSpot lead
        channel_id: Where to post the result
        thread_ts: If provided, post as thread reply (production mode)
        max_searches: Max web searches for enrichment
        include_lead_info: Include lead details in message (test mode)
        skip_research: If True, run triage only (no web research or assessment)

    Returns:
        ProcessedLead with results
    """
    # Step 1: Triage only — fast, no web searches
    triaged = triage_lead(settings, lead)

    # Step 2: React immediately so the outcome is visible before research starts
    if thread_ts:
        react_to_lead_message(
            settings,
            channel_id=channel_id,
            timestamp=thread_ts,
            is_promising=triaged.label.value == "promising",
        )

    # Step 3: Enrich promising leads (research + scoring) unless skipped
    if triaged.label.value == "promising" and not skip_research:
        classification: LeadClassification | EnrichedLeadClassification = enrich_lead(
            settings, lead, triaged, max_searches=max_searches
        )
    else:
        classification = triaged

    processed = ProcessedLead(
        lead=lead,
        classification=classification,
        slack_message=format_slack_message(lead, classification, include_lead_info=False),
    )

    post_to_slack(
        settings,
        processed,
        channel_id=channel_id,
        thread_ts=thread_ts,
        include_lead_info=include_lead_info,
    )

    return processed
