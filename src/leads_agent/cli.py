"""Rich CLI for leads-agent."""

from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from leads_agent.config import get_settings

app = typer.Typer(
    name="leads-agent",
    help="🧠 AI-powered Slack lead classifier",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def init(
    output: Path = typer.Option(
        Path(".env"),
        "--output",
        "-o",
        help="Path to write the .env file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing .env file",
    ),
):
    """Interactive setup wizard to create a .env configuration file."""
    rprint(Panel.fit("🚀 [bold cyan]Leads Agent Setup Wizard[/]", border_style="cyan"))

    if output.exists() and not force:
        if not Confirm.ask(f"[yellow]{output}[/] already exists. Overwrite?"):
            raise typer.Abort()

    rprint("\n[bold]Slack Configuration[/]")
    rprint("[dim]Create a Slack App at https://api.slack.com/apps[/]\n")

    slack_bot_token = Prompt.ask(
        "  [cyan]SLACK_BOT_TOKEN[/]",
        default="xoxb-...",
    )
    slack_signing_secret = Prompt.ask(
        "  [cyan]SLACK_SIGNING_SECRET[/]",
        default="",
    )
    slack_channel_id = Prompt.ask(
        "  [cyan]SLACK_CHANNEL_ID[/]",
        default="C...",
    )

    rprint("\n[bold]LLM Configuration[/]")
    rprint("[dim]Default uses OpenAI; set LLM_BASE_URL for Ollama/other providers[/]\n")

    openai_api_key = Prompt.ask(
        "  [cyan]OPENAI_API_KEY[/]",
        default="sk-...",
    )
    llm_model_name = Prompt.ask(
        "  [cyan]LLM_MODEL_NAME[/]",
        default="gpt-4o-mini",
    )

    rprint("\n[bold]Runtime Options[/]")
    dry_run = Confirm.ask("  [cyan]DRY_RUN[/] (don't post replies)?", default=True)

    env_content = f"""\
# Slack credentials
SLACK_BOT_TOKEN={slack_bot_token}
SLACK_SIGNING_SECRET={slack_signing_secret}
SLACK_CHANNEL_ID={slack_channel_id}

# LLM configuration (OpenAI by default)
OPENAI_API_KEY={openai_api_key}
LLM_MODEL_NAME={llm_model_name}
# Uncomment for Ollama or other OpenAI-compatible providers:
# LLM_BASE_URL=http://localhost:11434/v1

# Runtime
DRY_RUN={str(dry_run).lower()}
"""

    output.write_text(env_content)
    rprint(f"\n[green]✓[/] Configuration written to [bold]{output}[/]")
    rprint("[dim]Run [bold]leads-agent config[/] to verify settings[/]")


@app.command()
def config():
    """Display current configuration (from environment)."""
    try:
        settings = get_settings()
    except Exception as e:
        rprint(f"[red]Error loading settings:[/] {e}")
        raise typer.Exit(1)

    table = Table(title="Current Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("SLACK_BOT_TOKEN", _mask(settings.slack_bot_token))
    table.add_row("SLACK_SIGNING_SECRET", _mask(settings.slack_signing_secret))
    table.add_row("SLACK_CHANNEL_ID", settings.slack_channel_id or "[not set]")
    table.add_row("OPENAI_API_KEY", _mask(settings.openai_api_key))
    table.add_row("LLM_BASE_URL", settings.llm_base_url)
    table.add_row("LLM_MODEL_NAME", settings.llm_model_name)
    table.add_row("DRY_RUN", str(settings.dry_run))

    console.print(table)


def _mask(secret, visible: int = 4) -> str:
    """Mask a secret string, handling SecretStr or None."""
    if secret is None:
        return "[not set]"
    # Handle pydantic SecretStr
    val = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
    if len(val) <= visible:
        return "***"
    return val[:visible] + "*" * (len(val) - visible)


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the FastAPI server to receive Slack events."""
    import uvicorn

    rprint(Panel.fit("🚀 [bold green]Starting Leads Agent API[/]", border_style="green"))
    rprint(f"[dim]Listening on http://{host}:{port}/slack/events[/]\n")

    uvicorn.run(
        "leads_agent.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def backtest(
    limit: int = typer.Option(50, "--limit", "-n", help="Number of messages to fetch"),
):
    """Run classifier on historical Slack messages for testing."""
    from leads_agent.backtest import run_backtest

    rprint(Panel.fit("🔬 [bold magenta]Backtesting Lead Classifier[/]", border_style="magenta"))
    run_backtest(limit=limit)


@app.command("pull-history")
def pull_history(
    output: Path = typer.Option(
        Path("channel_history.json"),
        "--output",
        "-o",
        help="Output JSON file path",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Number of messages to fetch"),
    channel_id: str = typer.Option(None, "--channel", "-c", help="Channel ID (defaults to SLACK_CHANNEL_ID)"),
    print_only: bool = typer.Option(False, "--print", "-p", help="Print to console instead of saving to file"),
):
    """Fetch Slack channel history and save to JSON."""
    import json

    from slack_sdk.errors import SlackApiError

    from leads_agent.slack import slack_client

    settings = get_settings()
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


@app.command()
def classify(
    message: str = typer.Argument(..., help="Message text to classify"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show message history and agent trace"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full message content (no truncation)"),
):
    """Classify a single message (for quick testing)."""
    from leads_agent.llm import ClassificationResult, classify_message

    settings = get_settings()

    rprint(Panel.fit("🧠 [bold yellow]Classifying Message[/]", border_style="yellow"))
    rprint(f"[dim]{message}[/]\n")

    result = classify_message(settings, message, debug=debug)

    # Handle both return types
    if isinstance(result, ClassificationResult):
        classification = result.classification
        label_value = result.label
        confidence = result.confidence
        reason = result.reason
    else:
        classification = result
        label_value = result.label.value
        confidence = result.confidence
        reason = result.reason

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    label_color = {"spam": "red", "solicitation": "yellow", "promising": "green"}.get(label_value, "white")

    table.add_row("Label", f"[bold {label_color}]{label_value}[/]")
    table.add_row("Confidence", f"{confidence:.0%}")
    table.add_row("Reason", reason)

    # Show extracted contact info if present
    if classification.first_name or classification.last_name:
        name = f"{classification.first_name or ''} {classification.last_name or ''}".strip()
        table.add_row("Name", name)
    if classification.email:
        table.add_row("Email", classification.email)
    if classification.company:
        table.add_row("Company", classification.company)

    console.print(table)

    # Show debug info if requested
    if debug and isinstance(result, ClassificationResult):
        rprint("\n[bold cyan]─── Debug Info ───[/]")
        rprint(f"[dim]Token usage:[/] {result.usage}")
        rprint(f"\n[bold cyan]─── Message History ({len(result.message_history)} messages) ───[/]")
        rprint(f"[dim]{result.format_history(verbose=verbose)}[/]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
