from __future__ import annotations

import json
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request

from .config import Settings, get_settings
from .llm import classify_lead
from .models import HubSpotLead
from .slack import slack_client, verify_slack_request


def _is_hubspot_message(settings: Settings, event: dict[str, Any]) -> bool:
    """Check if event is a HubSpot bot message we should process."""
    if event.get("type") != "message":
        return False
    if settings.slack_channel_id and event.get("channel") != settings.slack_channel_id:
        return False
    # Must be a bot_message subtype
    if event.get("subtype") != "bot_message":
        return False
    # Must be from HubSpot
    if event.get("username", "").lower() != "hubspot":
        return False
    # Skip thread replies
    if event.get("thread_ts") and event.get("thread_ts") != event.get("ts"):
        return False
    # Must have attachments (where HubSpot puts lead data)
    if not event.get("attachments"):
        return False
    return True


def _handle_hubspot_lead(settings: Settings, event: dict[str, Any]) -> None:
    """Process a HubSpot lead message."""
    # Parse the HubSpot message
    lead = HubSpotLead.from_slack_event(event)
    if not lead:
        print("[SKIP] Could not parse HubSpot message")
        return

    print("\n[HUBSPOT LEAD]")
    print(f"  Name: {lead.first_name} {lead.last_name}")
    print(f"  Email: {lead.email}")
    print(f"  Company: {lead.company}")
    print(f"  Message: {lead.message[:100] if lead.message else 'N/A'}...")

    # Classify the lead
    result = classify_lead(settings, lead)

    print("\n[CLASSIFICATION]")
    print(f"  Label: {result.label.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Reason: {result.reason}")

    if settings.dry_run:
        print("[DRY RUN] Would post to Slack")
        return

    # Build response message
    label_emoji = {"spam": "🔴", "solicitation": "🟡", "promising": "🟢"}.get(result.label.value, "⚪")

    response_parts = [
        f"{label_emoji} *{result.label.value.upper()}* ({result.confidence:.0%})",
        f"_{result.reason}_",
    ]

    # Add extracted info if available
    if result.company and result.company != lead.company:
        response_parts.append(f"\n📋 Company: {result.company}")

    client = slack_client(settings)
    client.chat_postMessage(
        channel=event["channel"],
        thread_ts=event["ts"],
        text="\n".join(response_parts),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Leads Agent", description="AI-powered lead classification bot")

    @app.on_event("startup")
    async def startup():
        print("[STARTUP] Routes registered:")
        for route in app.routes:
            if hasattr(route, "methods"):
                print(f"  {route.methods} {route.path}")

    @app.get("/")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "service": "leads-agent"}

    @app.post("/slack/events")
    async def slack_events(req: Request, background: BackgroundTasks):
        body = await req.body()

        # Log incoming request for debugging
        sig = req.headers.get("X-Slack-Signature", "MISSING")
        ts = req.headers.get("X-Slack-Request-Timestamp", "MISSING")
        print("\n[SLACK] Incoming request")
        print(f"  Headers: X-Slack-Signature={sig[:20] if sig else 'NONE'}...")
        print(f"  Headers: X-Slack-Request-Timestamp={ts}")

        if not verify_slack_request(settings, req, body):
            print("  [ERROR] Signature verification FAILED")
            print(f"  Signing secret configured: {settings.slack_signing_secret is not None}")
            return {"error": "Invalid request"}

        print("  [OK] Signature verified")

        try:
            payload = await req.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}

        # Slack URL verification
        if payload.get("type") == "url_verification":
            print("  [OK] URL verification challenge received")
            return {"challenge": payload.get("challenge")}

        event = payload.get("event", {}) or {}

        # Always ack quickly; do work async
        if _is_hubspot_message(settings, event):
            print("  [OK] HubSpot lead detected, processing...")
            background.add_task(_handle_hubspot_lead, settings, event)
        else:
            print(f"  [SKIP] Not a HubSpot message (subtype={event.get('subtype')}, username={event.get('username')})")

        return {"ok": True}

    return app


app = create_app()
