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


def run_test_mode(
    settings: Settings | None = None,
    test_channel: str | None = None,
    max_searches: int = 4,
) -> None:
    """
    Start Socket Mode but post results to test channel instead of thread replies.

    Like production mode, but posts to a separate channel for testing.
    """
    settings = settings or get_settings()
    settings.require_slack_socket_mode()

    target_channel = test_channel or settings.slack_test_channel_id
    if not target_channel:
        raise ValueError("No test channel configured")

    app = App(
        token=settings.slack_bot_token.get_secret_value(),
    )

    @app.event("message")
    def handle_message(event: dict, say: "Say", client: "WebClient"):
        """Handle incoming messages - post to test channel."""
        if not _is_hubspot_message(settings, event):
            return

        channel = event.get("channel", "unknown")
        logger.info(f"HubSpot lead detected in {channel}")

        lead = HubSpotLead.from_slack_event(event)
        if not lead:
            logger.warning("Could not parse HubSpot message")
            return

        logger.info(f"Processing lead: {lead.first_name} {lead.last_name} <{lead.email}>")

        # Process and post to TEST channel (not as thread reply)
        with logfire.span(
            "bolt.test_mode",
            source_channel=channel,
            test_channel=target_channel,
            lead_email=lead.email,
        ):
            result = process_and_post(
                settings,
                lead,
                channel_id=target_channel,  # Post to test channel
                thread_ts=None,  # Not as a thread reply
                max_searches=max_searches,
                include_lead_info=True,  # Include lead details
            )

            logger.info(f"Classified: {result.label} ({result.classification.confidence:.0%})")
            if not settings.dry_run:
                logger.info(f"Posted to test channel: {target_channel}")

    @app.event({"type": "message", "subtype": "message_changed"})
    def handle_message_changed(event: dict):
        pass

    @app.event({"type": "message", "subtype": "message_deleted"})
    def handle_message_deleted(event: dict):
        pass

    handler = SocketModeHandler(
        app,
        settings.slack_app_token.get_secret_value(),
    )

    print("\n[STARTUP] Leads Agent - TEST MODE (Socket Mode)")
    print(f"  Listening on: {settings.slack_channel_id or 'all channels'}")
    print(f"  Posting to: {target_channel}")
    print(f"  Dry run: {settings.dry_run}")
    print("\nWaiting for HubSpot messages... (Ctrl+C to stop)\n")

    handler.start()


def collect_events(
    settings: Settings | None = None,
    keep: int = 20,
    output_file: str = "collected_events.json",
) -> None:
    """
    Collect raw Socket Mode events for debugging/inspection.

    Saves the complete raw payload for each event to a JSON file.
    Stops after collecting `keep` events or on Ctrl+C.
    """
    import json
    from pathlib import Path

    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse

    settings = settings or get_settings()
    settings.require_slack_socket_mode()

    collected: list[dict] = []

    def save_events():
        Path(output_file).write_text(json.dumps(collected, indent=2, default=str))
        print(f"\n[SAVED] {len(collected)} events to {output_file}")

    def handle_socket_mode_request(client: SocketModeClient, req: SocketModeRequest):
        """Capture every raw Socket Mode request."""
        # Acknowledge immediately
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        # Save the raw payload exactly as received
        collected.append(req.payload)
        print(f"[{len(collected)}/{keep}] type={req.type}")

        if len(collected) >= keep:
            save_events()
            print("\n[DONE] Reached target count.")
            import os
            os._exit(0)

    client = SocketModeClient(
        app_token=settings.slack_app_token.get_secret_value(),
        web_client=None,
    )
    client.socket_mode_request_listeners.append(handle_socket_mode_request)

    print("\n[COLLECT] Listening for raw Socket Mode events")
    print(f"  Target: {keep} events")
    print(f"  Output: {output_file}")
    print("\nWaiting for events... (Ctrl+C to stop early)\n")

    try:
        client.connect()
        from time import sleep
        while True:
            sleep(1)
    except KeyboardInterrupt:
        save_events()
        print("\n[INTERRUPTED] Saved partial collection.")
    finally:
        client.close()
