from __future__ import annotations

from collections.abc import Iterable

from .config import Settings, get_settings
from .llm import classify_message
from .slack import slack_client


def fetch_historical_messages(settings: Settings, limit: int = 200) -> Iterable[dict]:
    settings.require_slack()
    if settings.slack_channel_id is None:
        return []

    client = slack_client(settings)
    resp = client.conversations_history(channel=settings.slack_channel_id, limit=limit)

    for msg in resp.get("messages", []):
        if msg.get("subtype"):
            continue
        if msg.get("thread_ts"):
            continue
        if not msg.get("text"):
            continue
        yield msg


def run_backtest(settings: Settings | None = None, limit: int = 50) -> None:
    if settings is None:
        settings = get_settings()

    print(f"Backtesting last {limit} messages\n")

    for msg in fetch_historical_messages(settings, limit=limit):
        text = msg.get("text", "")
        result = classify_message(settings, text)

        print("-" * 60)
        print(text)
        print(f"\u2192 {result.label} ({result.confidence:.2f})")
        print(f"Reason: {result.reason}")
