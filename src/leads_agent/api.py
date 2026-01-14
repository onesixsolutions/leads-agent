from __future__ import annotations

import json
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request

from .config import Settings, get_settings
from .llm import classify_message
from .slack import slack_client, verify_slack_request


def _is_relevant_slack_message_event(settings: Settings, event: dict[str, Any]) -> bool:
    if event.get("type") != "message":
        return False
    if settings.slack_channel_id and event.get("channel") != settings.slack_channel_id:
        return False
    if event.get("subtype"):
        return False
    if event.get("thread_ts"):
        return False
    if not str(event.get("text", "")).strip():
        return False
    return True


def _handle_message_event(settings: Settings, event: dict[str, Any]) -> None:
    text = str(event.get("text", "")).strip()
    result = classify_message(settings, text)

    print("\nLIVE EVENT")
    print(text)
    print(result)

    if settings.dry_run:
        return

    client = slack_client(settings)
    client.chat_postMessage(
        channel=event["channel"],
        thread_ts=event["ts"],
        text=f"\U0001f9e0 Lead classification: *{result.label}* ({result.confidence:.2f})\n_{result.reason}_",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI()

    @app.post("/slack/events")
    async def slack_events(req: Request, background: BackgroundTasks):
        body = await req.body()
        if not verify_slack_request(settings, req, body):
            return {"error": "Invalid request"}

        try:
            payload = await req.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}

        # Slack URL verification
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}

        event = payload.get("event", {}) or {}

        # Always ack quickly; do work async
        if _is_relevant_slack_message_event(settings, event):
            background.add_task(_handle_message_event, settings, event)

        return {"ok": True}

    return app


app = create_app()
