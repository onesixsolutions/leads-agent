"""
Slack Bolt app for leads-agent using Socket Mode.

Socket Mode uses outbound WebSocket connections, eliminating the need
for a public HTTPS endpoint. Just set SLACK_BOT_TOKEN and SLACK_APP_TOKEN.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import logfire
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from .config import Settings, get_settings
from .models import HubSpotLead
from .processor import process_and_post

if TYPE_CHECKING:
    from slack_bolt.context.say import Say
    from slack_sdk import WebClient

# Configure logfire
logfire.configure()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_hubspot_message(settings: Settings, event: dict) -> bool:
    """Check if event is a HubSpot bot message we should process."""
    # Must be a bot_message subtype
    if event.get("subtype") != "bot_message":
        return False
    # Must be from HubSpot
    if event.get("username", "").lower() != "hubspot":
        return False
    # Skip thread replies (only process top-level messages)
    if event.get("thread_ts") and event.get("thread_ts") != event.get("ts"):
        return False
    # Must have attachments (where HubSpot puts lead data)
    if not event.get("attachments"):
        return False
    # Filter by channel if configured
    if settings.slack_channel_id and event.get("channel") != settings.slack_channel_id:
        return False
    return True


def create_bolt_app(settings: Settings | None = None) -> App:
    """
    Create and configure the Bolt app.

    Args:
        settings: Application settings. If None, loads from environment.

    Returns:
        Configured Bolt App instance.
    """
    settings = settings or get_settings()
    settings.require_slack_socket_mode()

    app = App(
        token=settings.slack_bot_token.get_secret_value(),
        # No signing_secret needed for Socket Mode
    )

    @app.event("message")
    def handle_message(event: dict, say: "Say", client: "WebClient"):
        """Handle incoming messages - filter for HubSpot leads."""
        if not _is_hubspot_message(settings, event):
            return

        channel = event.get("channel", "unknown")
        logger.info(f"HubSpot lead detected in {channel}")

        lead = HubSpotLead.from_slack_event(event)
        if not lead:
            logger.warning("Could not parse HubSpot message")
            return

        logger.info(f"Processing lead: {lead.first_name} {lead.last_name} <{lead.email}>")

        # Process and post (reuse existing logic)
        with logfire.span(
            "bolt.handle_hubspot_lead",
            channel=channel,
            thread_ts=event.get("ts"),
            lead_email=lead.email,
        ):
            result = process_and_post(
                settings,
                lead,
                channel_id=channel,
                thread_ts=event["ts"],
            )

            logger.info(f"Classified: {result.label} ({result.classification.confidence:.0%})")

    @app.event({"type": "message", "subtype": "message_changed"})
    def handle_message_changed(event: dict):
        """Ignore message edits."""
        pass

    @app.event({"type": "message", "subtype": "message_deleted"})
    def handle_message_deleted(event: dict):
        """Ignore message deletions."""
        pass

    return app


def run_socket_mode(settings: Settings | None = None) -> None:
    """
    Start the Bolt app in Socket Mode.

    This blocks until interrupted (Ctrl+C).
    """
    settings = settings or get_settings()
    settings.require_slack_socket_mode()

    app = create_bolt_app(settings)
    handler = SocketModeHandler(
        app,
        settings.slack_app_token.get_secret_value(),
    )

    print("\n[STARTUP] Leads Agent (Socket Mode)")
    print(f"  Channel filter: {settings.slack_channel_id or 'all channels bot is in'}")
    print(f"  Dry run: {settings.dry_run}")
    print("\nListening for HubSpot messages... (Ctrl+C to stop)\n")

    handler.start()
