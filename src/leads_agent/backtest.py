from __future__ import annotations

from collections.abc import Iterable

from .config import Settings, get_settings
from .agent import ClassificationResult, classify_lead
from .models import EnrichedLeadClassification, HubSpotLead
from .slack import slack_client


def fetch_hubspot_leads(settings: Settings, limit: int = 200) -> Iterable[tuple[dict, HubSpotLead]]:
    """Fetch historical HubSpot lead messages from Slack."""
    settings.require_slack_client()
    if settings.slack_channel_id is None:
        return []

    client = slack_client(settings)
    resp = client.conversations_history(channel=settings.slack_channel_id, limit=limit)

    for msg in resp.get("messages", []):
        # Only process HubSpot bot messages
        if msg.get("subtype") != "bot_message":
            continue
        if msg.get("username", "").lower() != "hubspot":
            continue
        # Skip thread replies
        if msg.get("thread_ts") and msg.get("thread_ts") != msg.get("ts"):
            continue

        # Parse the lead
        lead = HubSpotLead.from_slack_event(msg)
        if lead:
            yield msg, lead


def run_backtest(
    settings: Settings | None = None,
    limit: int = 50,
    max_searches: int = 4,
    debug: bool = False,
    verbose: bool = False,
) -> None:
    """Run classification on historical HubSpot leads."""
    if settings is None:
        settings = get_settings()

    modes = []
    if debug:
        modes.append("debug")
    mode_str = f" ({', '.join(modes)})" if modes else ""
    print(f"Backtesting last {limit} HubSpot leads{mode_str}\n")

    count = 0
    for msg, lead in fetch_hubspot_leads(settings, limit=limit):
        count += 1
        print("=" * 60)
        print(f"[{count}] Processing lead...")

        if debug:
            print(f"    Input: {lead.first_name} {lead.last_name} <{lead.email}>")
            if lead.company:
                print(f"    Company: {lead.company}")

        result = classify_lead(settings, lead, max_searches=max_searches, debug=debug)

        # Handle ClassificationResult wrapper when debug=True
        if isinstance(result, ClassificationResult):
            classification = result.classification
            label_value = result.label
            confidence = result.confidence
            reason = result.reason

            if debug:
                print(f"\n    Token usage: {result.usage}")
                print(f"    Messages exchanged: {len(result.message_history)}")
                if verbose:
                    print("\n    --- Message History ---")
                    print(result.format_history(verbose=True))
                else:
                    # Show condensed history - just tool calls
                    for i, msg in enumerate(result.message_history):
                        if hasattr(msg, "parts"):
                            for part in msg.parts:
                                if hasattr(part, "tool_name"):
                                    args_str = str(getattr(part, "args", {}))
                                    if len(args_str) > 80:
                                        args_str = args_str[:80] + "..."
                                    print(f"    🔧 {part.tool_name}: {args_str}")
        else:
            classification = result
            label_value = result.label.value
            confidence = result.confidence
            reason = result.reason

        label_emoji = {"ignore": "🚫", "promising": "✅"}.get(label_value, "❓")

        print()
        print(f"Name: {lead.first_name} {lead.last_name}")
        print(f"Email: {lead.email}")
        if lead.company:
            print(f"Company: {lead.company}")
        if lead.message:
            msg_preview = lead.message[:200] + "..." if len(lead.message) > 200 else lead.message
            print(f"Message: {msg_preview}")
        print()
        label_display = label_value.upper() if isinstance(label_value, str) else label_value
        print(f"{label_emoji} {label_display} ({confidence:.0%})")
        print(f"Reason: {reason}")
        if hasattr(classification, "score"):
            try:
                print(f"Score: {classification.score}/5 ({classification.action.value})")
                print(f"Score Reason: {classification.score_reason}")
            except Exception:
                pass
        if getattr(classification, "lead_summary", None):
            print(f"Summary: {classification.lead_summary}")
        if getattr(classification, "key_signals", None):
            print(f"Signals: {', '.join(classification.key_signals)}")
        if classification.company:
            print(f"Extracted Company: {classification.company}")

        # Show enrichment results if available
        if isinstance(classification, EnrichedLeadClassification):
            if classification.company_research:
                print("\n📊 Company Research:")
                cr = classification.company_research
                print(f"   {cr.company_name}: {cr.company_description}")
                if cr.industry:
                    print(f"   Industry: {cr.industry}")
                if cr.website:
                    print(f"   Website: {cr.website}")

            if classification.contact_research:
                print("\n👤 Contact Research:")
                cr = classification.contact_research
                if cr.title:
                    print(f"   {cr.full_name} - {cr.title}")
                if cr.linkedin_summary:
                    print(f"   {cr.linkedin_summary[:200]}...")

            if classification.research_summary:
                print(f"\n📝 Summary: {classification.research_summary}")

    print("=" * 60)
    if count == 0:
        print("No HubSpot leads found in channel history.")
        print("Make sure the bot is invited to the channel and HubSpot is posting there.")
    else:
        print(f"\nProcessed {count} leads.")
