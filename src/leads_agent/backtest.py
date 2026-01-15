from __future__ import annotations

from collections.abc import Iterable

from .config import Settings, get_settings
from .llm import classify_lead
from .models import HubSpotLead
from .slack import slack_client


def fetch_hubspot_leads(settings: Settings, limit: int = 200) -> Iterable[tuple[dict, HubSpotLead]]:
    """Fetch historical HubSpot lead messages from Slack."""
    settings.require_slack()
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


def run_backtest(settings: Settings | None = None, limit: int = 50) -> None:
    """Run classification on historical HubSpot leads."""
    if settings is None:
        settings = get_settings()

    print(f"Backtesting last {limit} HubSpot leads\n")

    count = 0
    for msg, lead in fetch_hubspot_leads(settings, limit=limit):
        count += 1
        result = classify_lead(settings, lead)

        label_emoji = {"spam": "🔴", "solicitation": "🟡", "promising": "🟢"}.get(result.label.value, "⚪")

        print("-" * 60)
        print(f"Name: {lead.first_name} {lead.last_name}")
        print(f"Email: {lead.email}")
        if lead.company:
            print(f"Company: {lead.company}")
        if lead.message:
            print(f"Message: {lead.message[:200]}...")
        print()
        print(f"{label_emoji} {result.label.value.upper()} ({result.confidence:.0%})")
        print(f"Reason: {result.reason}")
        if result.company:
            print(f"Extracted Company: {result.company}")

    if count == 0:
        print("No HubSpot leads found in channel history.")
        print("Make sure the bot is invited to the channel and HubSpot is posting there.")
