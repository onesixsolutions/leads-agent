import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from rich import print as rprint

from leads_agent.agent import ClassificationResult, classify_lead
from leads_agent.config import Settings, get_settings
from leads_agent.core.history import pull_history
from leads_agent.core.icp_report import format_icp_report
from leads_agent.models import EnrichedLeadClassification, HubSpotLead


def load_events_from_file(file_path: str | Path) -> list[dict]:
    """Load raw events from a JSON file created by `collect`."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    return data


def extract_leads_from_events(events: list[dict]) -> Iterable[tuple[dict, HubSpotLead]]:
    """
    Extract HubSpot leads from collected events.

    Handles both raw Socket Mode payloads and webhook-style events.
    Supports both old format (just payload) and new format (with type/envelope_id/payload).
    """
    for event_record in events:
        # Handle new format: {type, envelope_id, payload, ...}
        if "payload" in event_record and "type" in event_record:
            payload = event_record["payload"]
        else:
            # Old format: just the payload directly
            payload = event_record

        # Socket Mode payload has event nested under "event" key
        event = payload.get("event", payload)

        # Skip non-message events
        if event.get("type") != "message":
            continue

        # Only process HubSpot bot messages
        if event.get("subtype") != "bot_message":
            continue
        if event.get("username", "").lower() != "hubspot":
            continue
        # Skip thread replies
        if event.get("thread_ts") and event.get("thread_ts") != event.get("ts"):
            continue

        # Parse the lead
        lead = HubSpotLead.from_slack_event(event)
        if lead:
            yield event, lead


def list_leads(events: list[dict]) -> None:
    """
    Print an indexed table of the leads in `events` without classifying them.

    Costs nothing — no LLM calls, no searches. The printed index is what
    `--index` selects, and the timestamp is what `--ts` selects.
    """
    rows = list(extract_leads_from_events(events))
    if not rows:
        print("No HubSpot leads found.")
        return

    print(f"{'#':<5}{'Date':<18}{'Name':<24}{'Email':<36}Message")
    print("-" * 120)
    for i, (event, lead) in enumerate(rows, start=1):
        ts = event.get("ts", "")
        try:
            # Slack ts is epoch seconds; show it in the operator's local time.
            when = (
                datetime.fromtimestamp(float(ts), tz=UTC)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except (TypeError, ValueError):
            when = "?"
        name = f"{lead.first_name or ''} {lead.last_name or ''}".strip() or "(no name)"
        preview = " ".join((lead.message or lead.raw_text or "").split())[:52]
        print(f"{i:<5}{when:<18}{name[:23]:<24}{(lead.email or '-')[:35]:<36}{preview}")
    print("-" * 120)
    print(f"{len(rows)} leads. Select with --index N (repeatable) or --ts <slack_ts>.")


def run_backtest(
    events_file: str | Path | None,
    settings: Settings | None = None,
    limit: int | None = None,
    max_searches: int = 4,
    debug: bool = False,
    verbose: bool = False,
    channel_id: str | None = None,
    indices: list[int] | None = None,
    timestamps: list[str] | None = None,
    list_only: bool = False,
) -> None:
    """
    Run classification on leads from a collected events file or Slack history.

    Args:
        events_file: Path to JSON file created by `leads-agent collect`
        settings: Application settings
        limit: Max number of leads to process (None = all)
        max_searches: Max web searches per lead
        debug: Show debug output
        verbose: Show full message history (with debug)
        channel_id: Slack channel ID to pull history from if no file is provided
        indices: 1-based lead positions to process (as shown by `list_only`)
        timestamps: Slack message timestamps to process (stable across pulls)
        list_only: Print the indexed lead list and return without classifying
    """
    if settings is None:
        settings = get_settings()

    # Load events from file or pull history
    if events_file is not None:
        events = load_events_from_file(events_file)
        print(f"Loaded {len(events)} events from {events_file}\n")
    else:
        events = pull_history(
            channel_id=channel_id,
            limit=limit,
            output=None,
            print_only=False,
        )
        source_channel = channel_id or settings.slack_channel_id or "unknown"
        print(f"Loaded {len(events)} messages from Slack ({source_channel})\n")

    if list_only:
        list_leads(events)
        return

    wanted_idx = set(indices or [])
    wanted_ts = set(timestamps or [])
    selecting = bool(wanted_idx or wanted_ts)

    modes = []
    if debug:
        modes.append("debug")
    mode_str = f" ({', '.join(modes)})" if modes else ""
    if selecting:
        picks = [f"#{i}" for i in sorted(wanted_idx)] + [f"ts={t}" for t in sorted(wanted_ts)]
        limit_str = f" (selected: {', '.join(picks)})"
    else:
        limit_str = f" (limit: {limit})" if limit else ""
    print(f"Backtesting HubSpot leads{mode_str}{limit_str}\n")

    count = 0
    seen = 0
    matched_idx: set[int] = set()
    matched_ts: set[str] = set()
    for event, lead in extract_leads_from_events(events):
        seen += 1
        lead_ts = str(event.get("ts", ""))

        if selecting:
            # Skip anything not explicitly asked for. `seen` is the same
            # 1-based index that `--list` prints.
            if seen in wanted_idx:
                matched_idx.add(seen)
            elif lead_ts in wanted_ts:
                matched_ts.add(lead_ts)
            else:
                continue
        elif limit and count >= limit:
            break

        count += 1
        print("=" * 60)
        print(f"[{seen}] Processing lead...")

        if debug:
            print(f"    Input: {lead.first_name} {lead.last_name} <{lead.email}>")
            if lead.company:
                print(f"    Company: {lead.company}")

        result = classify_lead(settings, lead, max_searches=max_searches, debug=debug)

        # Handle ClassificationResult wrapper when debug=True
        if isinstance(result, ClassificationResult):
            classification = result.classification
            label_value = result.label
            reason = result.reason

            if debug:
                print(f"\n    Token usage: {result.usage}")
                print(f"    Messages exchanged: {len(result.message_history)}")
                if verbose:
                    print("\n    --- Message History ---")
                    print(result.format_history(verbose=True))
                else:
                    # Show condensed history - just tool calls
                    for msg in result.message_history:
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
        print(f"{label_emoji} {label_display}")
        print(f"Reason: {reason}")

        report = format_icp_report(classification)
        if report.strip():
            rprint(report)
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
    if selecting:
        missing_idx = sorted(wanted_idx - matched_idx)
        missing_ts = sorted(wanted_ts - matched_ts)
        if missing_idx:
            print(f"[WARN] No lead at index: {', '.join(str(i) for i in missing_idx)} (found {seen} leads)")
        if missing_ts:
            print(f"[WARN] No lead with ts: {', '.join(missing_ts)}")

    if count == 0:
        if selecting:
            print("Nothing matched the selection. Run with --list to see available leads.")
        else:
            print("No HubSpot leads found in events file.")
            print("Make sure the file contains HubSpot bot messages.")
    else:
        print(f"\nProcessed {count} lead(s) of {seen} found.")
