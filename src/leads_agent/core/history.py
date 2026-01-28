from slack_sdk.errors import SlackApiError
from rich import print as rprint
import typer
from rich.panel import Panel
import json

from pathlib import Path

from leads_agent.slack import slack_client
from leads_agent.config import get_settings

def pull_history(channel_id: str | None, limit: int, output: Path, print_only: bool):
    settings = get_settings()
    try:
        settings.require_slack_client()
    except Exception as e:
        rprint(f"[red]Error loading Slack settings:[/] {e}")
        raise typer.Exit(1)

    client = slack_client(settings)

    target_channel = channel_id or settings.slack_channel_id
    if not target_channel:
        rprint("[red]Error:[/] No channel ID provided. Use --channel or set SLACK_CHANNEL_ID")
        raise typer.Exit(1)

    rprint(Panel.fit("📥 [bold blue]Fetching Channel History[/]", border_style="blue"))
    rprint(f"[dim]Channel: {target_channel} | Limit: {limit}[/]\n")

    try:
        resp = client.conversations_history(channel=target_channel, limit=limit)
    except SlackApiError as e:
        error_code = e.response.get("error", "unknown")
        rprint(f"[red]Slack API error:[/] {error_code}")

        # Provide helpful hints for common errors
        hints = {
            "not_in_channel": "The bot must be invited to the channel. Use /invite @bot-name in Slack.",
            "channel_not_found": "Check that the channel ID is correct.",
            "missing_scope": "The bot token needs 'channels:history' (public) or 'groups:history' (private) scope.",
            "invalid_auth": "The SLACK_BOT_TOKEN is invalid or expired.",
        }
        if error_code in hints:
            rprint(f"[yellow]Hint:[/] {hints[error_code]}")

        raise typer.Exit(1)

    messages = resp.get("messages", [])

    if print_only:
        for msg in messages:
            rprint("=" * 60)
            rprint(json.dumps(msg, indent=2))
    else:
        output.write_text(json.dumps(messages, indent=2))
        rprint(f"[green]✓[/] Saved {len(messages)} messages to [bold]{output}[/]")