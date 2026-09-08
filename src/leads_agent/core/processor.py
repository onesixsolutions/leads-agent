import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from slack_sdk.errors import SlackApiError

from leads_agent.agent import classify_lead, enrich_lead, triage_lead
from leads_agent.briefs import publish_brief
from leads_agent.icp_fit import CRITERIA, CriterionSpec, verdict_display
from leads_agent.models import (
    CriterionStatus,
    EnrichedLeadClassification,
    HubSpotLead,
    ICPCriterion,
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

# Slack refuses a section block over 3,000 characters, so the detailed brief is
# posted in chunks. The margin absorbs Slack's own framing.
SLACK_BLOCK_LIMIT = 3000
SLACK_CHUNK_LIMIT = 2800

# Card budget. Slack collapses anything past ~4,000 characters behind "Show
# more", so a card that has to be expanded has already failed. These caps keep
# a worst-case card near 1,000 characters.
CARD_MAX_CRITERIA = 3
CARD_STATEMENT_CHARS = 240
CARD_BULLET_CHARS = 140
CARD_TAKE_CHARS = 200
CARD_REASON_CHARS = 180


@dataclass
class ProcessedLead:
    """Result of processing a lead."""

    lead: HubSpotLead
    classification: LeadClassification | EnrichedLeadClassification
    slack_message: str
    # The detailed brief that goes in the thread beneath the card. Defaults to
    # empty so older construction sites keep working.
    full_brief: str = ""

    @property
    def label(self) -> str:
        return self.classification.label.value

    @property
    def is_promising(self) -> bool:
        return self.label == "promising"


# --- Text helpers ---------------------------------------------------------


def _clip(text: str | None, limit: int) -> str:
    """
    Shorten to `limit` characters, preferring a sentence boundary.

    A card is only worth reading if it reads cleanly, and a cut mid-word looks
    like a bug rather than a summary.
    """
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed

    window = collapsed[: limit + 1]
    sentence_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if sentence_end >= limit // 2:
        return window[: sentence_end + 1]

    space = window.rfind(" ")
    stem = window[:space] if space > 0 else window[:limit]
    return stem.rstrip(" ,;:-—") + "…"


def _split_long_line(line: str, limit: int) -> list[str]:
    """Break a single over-long line on word boundaries so chunking can't fail."""
    if len(line) <= limit:
        return [line]

    pieces: list[str] = []
    remaining = line
    while len(remaining) > limit:
        window = remaining[:limit]
        space = window.rfind(" ")
        cut = space if space > limit // 2 else limit
        pieces.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_for_slack(text: str, limit: int = SLACK_CHUNK_LIMIT) -> list[str]:
    """
    Split text into Slack-postable chunks, breaking on line boundaries.

    Returns an empty list for empty text so callers can post unconditionally.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for raw_line in text.split("\n"):
        for line in _split_long_line(raw_line, limit):
            cost = len(line) + (1 if current else 0)
            if current and size + cost > limit:
                chunks.append("\n".join(current))
                current, size, cost = [], 0, len(line)
            current.append(line)
            size += cost

    if current:
        chunks.append("\n".join(current))
    return chunks


# --- Decision card --------------------------------------------------------


def _criterion_bullet(spec: CriterionSpec, criterion: ICPCriterion) -> str:
    finding = _clip(criterion.finding, CARD_BULLET_CHARS)
    return f"• *{spec.label}* — {finding}" if finding else f"• *{spec.label}*"


def _deciding_criteria(
    classification: LeadClassification | EnrichedLeadClassification,
) -> tuple[str, list[str]]:
    """
    The criteria that actually produced the verdict: a header and its bullets.

    Only these belong on the card. Reporting all eleven every time is what
    turned the old message into a wall of text nobody expanded. CRITERIA order
    puts the hard gates first, so truncating the tail drops the least important
    lines.
    """
    verdict = getattr(classification, "icp_verdict", None)
    assessment = getattr(classification, "icp_assessment", None)
    if assessment is None:
        return "", []

    if verdict == ICPVerdict.out_of_icp:
        header, wanted = "*Fails:*", (CriterionStatus.not_met,)
    elif verdict in (ICPVerdict.needs_verification, ICPVerdict.partial_fit):
        header, wanted = "*Unverified:*", (CriterionStatus.unknown,)
    elif verdict == ICPVerdict.in_icp:
        header, wanted = "*Qualifies:*", (CriterionStatus.met, CriterionStatus.partial)
    else:
        return "", []

    bullets: list[str] = []
    for spec in CRITERIA:
        # Only gates decide a verdict; the rest is context for the full brief.
        if not spec.is_gate:
            continue
        criterion = getattr(assessment, spec.field, None)
        if criterion is None or criterion.status not in wanted:
            continue
        bullets.append(_criterion_bullet(spec, criterion))

    return header, bullets


def _lead_identity(lead: HubSpotLead) -> str:
    name = f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "Unknown"
    email = lead.email
    parts = [f"*{name}*"]
    if email:
        parts.append(f"<mailto:{email}|{email}>")
    if lead.company:
        parts.append(lead.company)
    return " · ".join(parts)


def format_slack_message(
    lead: HubSpotLead,
    classification: LeadClassification | EnrichedLeadClassification,
    include_lead_info: bool = False,
    *,
    brief_url: str | None = None,
) -> str:
    """
    Format the decision card: does this lead deserve anyone's time, and why?

    Deliberately short (~10 lines). Everything else — all eleven criteria,
    risks, talking points, research — lives in `format_full_brief` and is
    posted as a thread reply.

    Args:
        lead: The parsed lead data
        classification: The classification result
        include_lead_info: If True, prepend a one-line lead identity header
        brief_url: Link to the hosted full brief, rendered when present
    """
    parts: list[str] = []

    if include_lead_info:
        parts.append(_lead_identity(lead))

    verdict = getattr(classification, "icp_verdict", None)
    has_verdict = verdict is not None and verdict != ICPVerdict.not_evaluated

    if not has_verdict:
        # No ICP verdict, so triage is the whole story (spam, or research
        # skipped). When a verdict follows, restating "GO" wastes a line.
        if classification.label.value == "promising":
            parts.append("✅ *GO* — genuine inquiry")
        else:
            parts.append("🚫 *IGNORE* — not a genuine inquiry")
        parts.append(f"_{_clip(classification.reason, CARD_REASON_CHARS)}_")
    else:
        emoji, label = verdict_display(verdict)
        parts.append(f"{emoji} *{label}*")

        brief = getattr(classification, "brief", None)
        if brief and brief.icp_statement:
            parts.append(f"_{_clip(brief.icp_statement, CARD_STATEMENT_CHARS)}_")

        header, bullets = _deciding_criteria(classification)
        if bullets:
            parts.append("")
            parts.append(header)
            parts.extend(bullets[:CARD_MAX_CRITERIA])
            overflow = len(bullets) - CARD_MAX_CRITERIA
            if overflow > 0:
                parts.append(f"_+{overflow} more in the brief_")

        if brief and brief.analyst_take:
            parts.append("")
            parts.append(f"🧠 {_clip(brief.analyst_take, CARD_TAKE_CHARS)}")

    footer: list[str] = []
    action = getattr(classification, "action", None)
    if action is not None:
        footer.append(f"*Action:* `{action.value}`")
    if brief_url:
        footer.append(f"<{brief_url}|Full brief →>")
    if footer:
        parts.append("")
        parts.append(" · ".join(footer))

    return "\n".join(parts)


# --- Full brief -----------------------------------------------------------


def has_full_brief(classification: LeadClassification | EnrichedLeadClassification) -> bool:
    """
    Whether there is detail worth a thread reply.

    Spam and triage-only leads have nothing the card doesn't already say.
    """
    verdict = getattr(classification, "icp_verdict", None)
    return bool(
        (verdict is not None and verdict != ICPVerdict.not_evaluated)
        or getattr(classification, "company_research", None)
        or getattr(classification, "contact_research", None)
        or getattr(classification, "research_summary", None)
    )


def format_full_brief(
    lead: HubSpotLead,
    classification: LeadClassification | EnrichedLeadClassification,
    include_lead_info: bool = False,
) -> str:
    """
    Render the detailed brief: every reason, the judgement layer and research.

    May exceed Slack's 3,000-character block limit — pass the result through
    `chunk_for_slack` before posting. Returned whole so other consumers (e.g. a
    hosted HTML brief) can use it as one document.
    """
    parts: list[str] = []

    if include_lead_info:
        parts.append(f"*Lead:* {_lead_identity(lead)}")
        if lead.message:
            msg_preview = lead.message[:300] + "..." if len(lead.message) > 300 else lead.message
            parts.append(f"*Message:* {msg_preview}")
        parts.append("")

    # The card above already carries the triage line and the verdict. Repeating
    # them here put "✅ GO" directly beneath "⛔ NOT IN ICP", which reads as a
    # contradiction; the brief starts where the card stops.

    # ICP verdict detail: the criteria that produced it.
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
            contact = classification.contact_research
            parts.append("\n*👤 Contact Research:*")
            title_str = f" - {contact.title}" if contact.title else ""
            parts.append(f"• *{contact.full_name}*{title_str}")
            if contact.linkedin_summary:
                summary = contact.linkedin_summary
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                parts.append(f"• {summary}")
            if contact.relevance_notes:
                parts.append(f"• Relevance: {contact.relevance_notes}")

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
        ProcessedLead with classification, decision card and full brief
    """
    if skip_research:
        classification: LeadClassification | EnrichedLeadClassification = triage_lead(settings, lead)
    else:
        classification = classify_lead(settings, lead, max_searches=max_searches)
        # Handle ClassificationResult wrapper (from debug mode)
        if hasattr(classification, "classification"):
            classification = classification.classification

    return ProcessedLead(
        lead=lead,
        classification=classification,
        slack_message=format_slack_message(lead, classification, include_lead_info=False),
        full_brief=format_full_brief(lead, classification, include_lead_info=False),
    )


def post_to_slack(
    settings: "Settings",
    processed: ProcessedLead,
    *,
    channel_id: str,
    thread_ts: str | None = None,
    include_lead_info: bool = False,
    brief_url: str | None = None,
) -> None:
    """
    Post the decision card, then the full brief as a reply beneath it.

    Args:
        settings: Application settings
        processed: The processed lead result
        channel_id: Slack channel ID to post to
        thread_ts: If provided, post as thread reply; otherwise post to main channel
        include_lead_info: If True, include lead details in message
        brief_url: Link to the hosted full brief, rendered on the card
    """
    if settings.dry_run:
        print(f"[DRY RUN] Would post to {channel_id}" + (f" (thread: {thread_ts})" if thread_ts else ""))
        return

    # Re-format only when the card depends on arguments the cached one lacked.
    card = (
        format_slack_message(
            processed.lead,
            processed.classification,
            include_lead_info=include_lead_info,
            brief_url=brief_url,
        )
        if include_lead_info or brief_url
        else processed.slack_message
    )

    client = slack_client(settings)

    kwargs = {
        "channel": channel_id,
        "text": card,
    }
    if thread_ts:
        kwargs["thread_ts"] = thread_ts

    response = client.chat_postMessage(**kwargs)

    if not has_full_brief(processed.classification):
        return

    # The hosted brief is the document; the card links to it. Posting the same
    # content again as thread chunks would duplicate it and, at four messages
    # per lead, drown the thread. Thread text is the fallback for when briefs
    # are not configured.
    if brief_url:
        return

    detail = processed.full_brief or format_full_brief(
        processed.lead, processed.classification, include_lead_info=include_lead_info
    )
    # In production the card is already a reply to the HubSpot message, so the
    # brief joins that thread; in test mode the card itself starts one.
    reply_ts = thread_ts or (response.get("ts") if response is not None else None)
    if not reply_ts:
        logger.warning("No thread timestamp available; skipping full brief for %s", channel_id)
        return

    for chunk in chunk_for_slack(detail):
        client.chat_postMessage(channel=channel_id, thread_ts=reply_ts, text=chunk)


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
    brief_url: str | None = None,
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
        brief_url: Link to the hosted full brief, rendered on the card

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

    # Step 4: publish the hosted brief so the card can link to it.
    #
    # Best-effort by design: publish_brief returns None rather than raising, so
    # a brief we could not store never costs us the Slack reply. Suppressed in
    # dry runs because writing to S3 is an outward side effect, and DRY_RUN is
    # the switch people rely on to have none.
    if brief_url is None and not settings.dry_run and has_full_brief(classification):
        try:
            ref = publish_brief(lead, classification, settings=settings)
        except Exception:
            # publish_brief already swallows storage errors, so reaching here
            # means something unforeseen. Belt and braces regardless: losing a
            # brief link is trivial, losing the lead reply is not.
            logger.exception("Brief publishing raised unexpectedly; posting without a link")
            ref = None
        if ref is not None:
            brief_url = ref.url

    processed = ProcessedLead(
        lead=lead,
        classification=classification,
        slack_message=format_slack_message(
            lead, classification, include_lead_info=False, brief_url=brief_url
        ),
        full_brief=format_full_brief(lead, classification, include_lead_info=include_lead_info),
    )

    post_to_slack(
        settings,
        processed,
        channel_id=channel_id,
        thread_ts=thread_ts,
        include_lead_info=include_lead_info,
        brief_url=brief_url,
    )

    return processed
