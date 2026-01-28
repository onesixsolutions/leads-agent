from slack_sdk.errors import SlackApiError
from rich import print as rprint
import typer
from rich.panel import Panel

from leads_agent.models import HubSpotLead
from leads_agent.core.processor import process_and_post
from leads_agent.slack import slack_client
from leads_agent.config import get_settings



def replay(channel_id: str, limit: int, dry_run: bool, max_searches: int):
    settings = get_settings()
    try:
        settings.require_slack_client()
    except Exception as e:
        rprint(f"[red]Error loading Slack settings:[/] {e}")
        raise typer.Exit(1)

    # Override dry_run if explicitly set
    if dry_run is not None:
        settings.dry_run = dry_run

    target_channel = channel_id or settings.slack_channel_id
    if not target_channel:
        rprint("[red]Error:[/] No channel ID provided. Use --channel or set SLACK_CHANNEL_ID")
        raise typer.Exit(1)

    if limit <= 0:
        rprint("[red]Error:[/] --limit must be >= 1")
        raise typer.Exit(1)

    client = slack_client(settings)

    rprint(Panel.fit("⏪ [bold blue]Replaying Channel History[/]", border_style="blue"))
    rprint(f"[dim]Channel: {target_channel} | Leads to replay: {limit} | Dry run: {settings.dry_run}[/]\n")

    # Paginate until we replay `limit` HubSpot lead messages (or history is exhausted).
    processed = 0
    cursor: str | None = None
    scanned = 0

    try:
        while processed < limit:
            history_kwargs: dict = {"channel": target_channel, "limit": 200}
            if cursor:
                history_kwargs["cursor"] = cursor

            resp = client.conversations_history(**history_kwargs)

            messages = resp.get("messages", [])
            scanned += len(messages)

            if not messages:
                break

            for msg in messages:
                # conversations_history messages don't include channel; add for parity with event payloads
                event = dict(msg)
                event["channel"] = target_channel

                # Quick filter (match production behavior)
                if event.get("subtype") != "bot_message":
                    continue
                if event.get("username", "").lower() != "hubspot":
                    continue
                if event.get("thread_ts") and event.get("thread_ts") != event.get("ts"):
                    continue
                if not event.get("attachments"):
                    continue

                lead = HubSpotLead.from_slack_event(event)
                if not lead:
                    continue

                processed += 1

                result = process_and_post(
                    settings,
                    lead,
                    channel_id=target_channel,
                    thread_ts=event.get("ts"),  # replay as thread reply, like production
                    max_searches=max_searches,
                )

                if settings.dry_run:
                    rprint(
                        Panel(
                            result.slack_message,
                            title=f"Replay {processed}/{limit}",
                            border_style="yellow",
                        )
                    )
                else:
                    ts = event.get("ts", "?")
                    rprint(f"[green]✓[/] Posted replay {processed}/{limit} (thread_ts={ts})")

                if processed >= limit:
                    break

            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break

    except SlackApiError as e:
        error_code = e.response.get("error", "unknown")
        rprint(f"[red]Slack API error:[/] {error_code}")

        hints = {
            "not_in_channel": "The bot must be invited to the channel. Use /invite @bot-name in Slack.",
            "channel_not_found": "Check that the channel ID is correct.",
            "missing_scope": "The bot token needs 'channels:history' (public) or 'groups:history' (private) scope.",
            "invalid_auth": "The SLACK_BOT_TOKEN is invalid or expired.",
        }
        if error_code in hints:
            rprint(f"[yellow]Hint:[/] {hints[error_code]}")
        raise typer.Exit(1)

    if processed == 0:
        rprint("[yellow]No HubSpot lead messages found in the scanned history.[/]")
        rprint("[dim]Make sure HubSpot messages are present and include attachments.[/]")
        rprint(f"[dim]Messages scanned: {scanned}[/]")
    elif processed < limit:
        rprint(f"\n[yellow]Replayed {processed}/{limit} lead messages (history exhausted).[/]")
        rprint(f"[dim]Messages scanned: {scanned}[/]")