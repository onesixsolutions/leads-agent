from __future__ import annotations

import json
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .config import Settings, get_settings
from .models import HubSpotLead
from .processor import process_and_post
from .prompts import ICPConfig, PromptConfig, get_prompt_manager
from .slack import verify_slack_request


class PromptConfigResponse(BaseModel):
    """Response model for prompt configuration endpoints."""

    config: PromptConfig
    classification_prompt: str
    research_prompt: str


class PromptConfigUpdate(BaseModel):
    """Request model for updating prompt configuration."""

    config: PromptConfig


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
    """Process a HubSpot lead message (production mode)."""
    lead = HubSpotLead.from_slack_event(event)
    if not lead:
        print("[SKIP] Could not parse HubSpot message")
        return

    print("\n[HUBSPOT LEAD]")
    print(f"  Name: {lead.first_name} {lead.last_name}")
    print(f"  Email: {lead.email}")
    print(f"  Company: {lead.company}")
    print(f"  Message: {lead.message[:100] if lead.message else 'N/A'}...")

    # Process and post as thread reply
    result = process_and_post(
        settings,
        lead,
        channel_id=event["channel"],
        thread_ts=event["ts"],  # Reply in thread
        enrich=False,  # Production mode doesn't enrich by default
    )

    print("\n[CLASSIFICATION]")
    print(f"  Label: {result.label}")
    print(f"  Confidence: {result.classification.confidence:.0%}")
    print(f"  Reason: {result.classification.reason}")


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

    # ─────────────────────────────────────────────────────────────────────────────
    # Prompt Configuration Endpoints
    # ─────────────────────────────────────────────────────────────────────────────

    @app.get("/config/prompts", response_model=PromptConfigResponse)
    async def get_prompt_config():
        """
        Get the current prompt configuration.

        Returns the configuration along with the fully-rendered prompts
        that will be sent to the LLM.
        """
        manager = get_prompt_manager()
        return PromptConfigResponse(
            config=manager.config,
            classification_prompt=manager.build_classification_prompt(),
            research_prompt=manager.build_research_prompt(),
        )

    @app.put("/config/prompts", response_model=PromptConfigResponse)
    async def update_prompt_config(update: PromptConfigUpdate):
        """
        Update the prompt configuration.

        This updates the runtime configuration. Changes are not persisted
        and will be lost on restart. To persist, save to prompt_config.json
        or set PROMPT_CONFIG_JSON environment variable.
        """
        manager = get_prompt_manager()
        manager.update_config(update.config)
        return PromptConfigResponse(
            config=manager.config,
            classification_prompt=manager.build_classification_prompt(),
            research_prompt=manager.build_research_prompt(),
        )

    @app.patch("/config/prompts", response_model=PromptConfigResponse)
    async def patch_prompt_config(update: dict[str, Any]):
        """
        Partially update the prompt configuration.

        Only the provided fields will be updated. Useful for updating
        specific aspects without providing the entire configuration.
        """
        manager = get_prompt_manager()
        current = manager.config.model_dump()

        # Deep merge the update
        def deep_merge(base: dict, updates: dict) -> dict:
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(current, update)

        try:
            new_config = PromptConfig.model_validate(merged)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

        manager.update_config(new_config)
        return PromptConfigResponse(
            config=manager.config,
            classification_prompt=manager.build_classification_prompt(),
            research_prompt=manager.build_research_prompt(),
        )

    @app.delete("/config/prompts")
    async def reset_prompt_config():
        """
        Reset the prompt configuration to defaults.

        Clears any runtime overrides and reloads from the base configuration
        (environment variables or config file).
        """
        manager = get_prompt_manager()
        manager.reset_config()
        return {
            "status": "reset",
            "message": "Prompt configuration reset to defaults",
        }

    @app.get("/config/prompts/preview")
    async def preview_prompt_config(
        company_name: str | None = None,
        services_description: str | None = None,
        icp_description: str | None = None,
        target_industries: str | None = None,
        target_company_sizes: str | None = None,
    ):
        """
        Preview prompts with temporary configuration.

        Useful for testing configuration changes before applying them.
        Does not modify the actual configuration.
        """
        # Build temporary config from query params
        icp = None
        if any([icp_description, target_industries, target_company_sizes]):
            icp = ICPConfig(
                description=icp_description,
                target_industries=target_industries.split(",") if target_industries else None,
                target_company_sizes=target_company_sizes.split(",") if target_company_sizes else None,
            )

        temp_config = PromptConfig(
            company_name=company_name,
            services_description=services_description,
            icp=icp,
        )

        # Create temporary manager
        from .prompts import PromptManager

        temp_manager = PromptManager(temp_config)

        return {
            "config": temp_config.model_dump(exclude_none=True),
            "classification_prompt": temp_manager.build_classification_prompt(),
            "research_prompt": temp_manager.build_research_prompt(),
        }

    return app


app = create_app()
