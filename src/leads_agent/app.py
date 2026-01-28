import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import logfire
from rich.console import Console
from rich.logging import RichHandler
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from leads_agent.config import Settings, get_settings
from leads_agent.models import HubSpotLead
from leads_agent.core.processor import process_and_post

if TYPE_CHECKING:
    from slack_bolt.context.say import Say
    from slack_sdk import WebClient

# Configure logfire only if token is available
_logfire_enabled = bool(os.environ.get("LOGFIRE_TOKEN"))
if _logfire_enabled:
    try:
        logfire.configure()
    except Exception:
        # If configuration fails, disable logfire
        _logfire_enabled = False


@contextmanager
def _logfire_span(name: str, **kwargs):
    """Context manager for logfire spans that works even when logfire is disabled."""
    if _logfire_enabled:
        with logfire.span(name, **kwargs):
            yield
    else:
        yield

# Set up logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)
console = Console()


def _is_hubspot_message(settings: Settings, event: dict) -> bool:
    """Check if event is a HubSpot bot message we should process."""
    if settings.debug:
        console.print("[bold cyan]Event:[/]")
        console.print(event)
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
        with _logfire_span(
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

    print("\n[STARTUP] Leads Agent")
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
        with _logfire_span(
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
    import threading
    from pathlib import Path
    from time import sleep

    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse

    settings = settings or get_settings()
    settings.require_slack_socket_mode()

    collected: list[dict] = []
    lock = threading.Lock()
    should_stop = threading.Event()

    def save_events():
        """Save collected events to file (thread-safe)."""
        with lock:
            if not collected:
                return
            try:
                Path(output_file).write_text(json.dumps(collected, indent=2, default=str))
                print(f"\n[SAVED] {len(collected)} events to {output_file}")
            except Exception as e:
                print(f"\n[ERROR] Failed to save events: {e}")

    def handle_socket_mode_request(client: SocketModeClient, req: SocketModeRequest):
        """Capture every raw Socket Mode request."""
        try:
            # Acknowledge immediately
            client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

            # Save the full request data (not just payload) for complete debugging
            # This includes type, envelope_id, and the full payload structure
            event_data = {
                "type": req.type,
                "envelope_id": req.envelope_id,
                "payload": req.payload,
                # Also include raw request attributes if available
                "raw_request": {
                    "retry_num": getattr(req, "retry_num", None),
                    "retry_reason": getattr(req, "retry_reason", None),
                } if hasattr(req, "retry_num") else None,
            }

            with lock:
                collected.append(event_data)
                count = len(collected)
                
            # Log with more detail
            event_type = req.type
            event_subtype = None
            if isinstance(req.payload, dict):
                event_data_inner = req.payload.get("event", {})
                if isinstance(event_data_inner, dict):
                    event_subtype = event_data_inner.get("subtype")
                    event_type_detail = event_data_inner.get("type")
                    if event_type_detail:
                        event_type = f"{event_type}/{event_type_detail}"
            
            print(f"[{count}/{keep}] type={event_type}" + (f" subtype={event_subtype}" if event_subtype else ""))

            # Save periodically (every 5 events) to avoid data loss
            if count % 5 == 0:
                save_events()

            # Check if we've reached the target
            if count >= keep:
                save_events()
                print("\n[DONE] Reached target count.")
                should_stop.set()
                return

        except Exception as e:
            print(f"\n[ERROR] Failed to handle request: {e}")
            import traceback
            traceback.print_exc()

    client = SocketModeClient(
        app_token=settings.slack_app_token.get_secret_value(),
        web_client=None,
    )
    client.socket_mode_request_listeners.append(handle_socket_mode_request)

    print("\n[COLLECT] Listening for raw Socket Mode events")
    print(f"  Target: {keep} events")
    print(f"  Output: {output_file}")
    print(f"  Auto-save: Every 5 events")
    print("\nWaiting for events... (Ctrl+C to stop early)\n")

    try:
        client.connect()
        # Wait until we should stop (either target reached or interrupted)
        while not should_stop.is_set():
            sleep(0.5)
            # Check connection health
            if not client.is_connected():
                print("\n[WARNING] Socket Mode connection lost. Reconnecting...")
                try:
                    client.connect()
                except Exception as e:
                    print(f"[ERROR] Failed to reconnect: {e}")
                    break
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving collected events...")
        save_events()
        print("\n[INTERRUPTED] Saved partial collection.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        save_events()
    finally:
        # Final save to ensure nothing is lost
        save_events()
        try:
            client.close()
        except Exception:
            pass
