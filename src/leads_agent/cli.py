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
    slack_test_channel_id = Prompt.ask(
        "  [cyan]SLACK_TEST_CHANNEL_ID[/] [dim](optional, for testing)[/]",
        default="",
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

    # Build env content
    env_lines = [
        "# Slack credentials",
        f"SLACK_BOT_TOKEN={slack_bot_token}",
        f"SLACK_SIGNING_SECRET={slack_signing_secret}",
        f"SLACK_CHANNEL_ID={slack_channel_id}",
    ]
    if slack_test_channel_id:
        env_lines.append(f"SLACK_TEST_CHANNEL_ID={slack_test_channel_id}")

    env_lines.extend(
        [
            "",
            "# LLM configuration (OpenAI by default)",
            f"OPENAI_API_KEY={openai_api_key}",
            f"LLM_MODEL_NAME={llm_model_name}",
            "# Uncomment for Ollama or other OpenAI-compatible providers:",
            "# LLM_BASE_URL=http://localhost:11434/v1",
            "",
            "# Runtime",
            f"DRY_RUN={str(dry_run).lower()}",
        ]
    )

    env_content = "\n".join(env_lines) + "\n"

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
    table.add_row("SLACK_TEST_CHANNEL_ID", settings.slack_test_channel_id or "[not set]")
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
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Research promising leads with web search"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches per lead"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show agent steps and token usage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full message history (with --debug)"),
):
    """Run classifier on historical HubSpot leads for testing."""
    from leads_agent.backtest import run_backtest

    modes = []
    if enrich:
        modes.append("enrichment")
    if debug:
        modes.append("debug")
    mode_str = f" [dim]({', '.join(modes)})[/]" if modes else ""
    title = f"🔬 [bold magenta]Backtesting Lead Classifier[/]{mode_str}"
    rprint(Panel.fit(title, border_style="magenta"))
    run_backtest(limit=limit, enrich=enrich, max_searches=max_searches, debug=debug, verbose=verbose)


@app.command()
def test(
    limit: int = typer.Option(5, "--limit", "-n", help="Number of leads to process"),
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Research promising leads with web search"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches per lead"),
    test_channel: str = typer.Option(None, "--channel", "-c", help="Test channel ID"),
    dry_run: bool = typer.Option(None, "--dry-run/--live", help="Override DRY_RUN config setting"),
):
    """
    Test mode: process historical leads and post results to a test channel.

    Pulls HubSpot leads from SLACK_CHANNEL_ID, processes them,
    and posts results to SLACK_TEST_CHANNEL_ID (not as threads).

    Respects DRY_RUN config setting. Use --dry-run or --live to override.
    """
    from leads_agent.backtest import fetch_hubspot_leads
    from leads_agent.processor import process_and_post

    settings = get_settings()

    # Override dry_run if explicitly set
    if dry_run is not None:
        settings.dry_run = dry_run

    # Determine test channel
    target_channel = test_channel or settings.slack_test_channel_id
    if not target_channel:
        rprint("[red]Error:[/] No test channel configured.")
        rprint("[dim]Set SLACK_TEST_CHANNEL_ID in .env or use --channel[/]")
        raise typer.Exit(1)

    rprint(Panel.fit("🧪 [bold cyan]Test Mode[/]", border_style="cyan"))
    rprint(f"[dim]Source: {settings.slack_channel_id} → Target: {target_channel}[/]")
    rprint(f"[dim]Limit: {limit} | Enrich: {enrich} | Dry run: {settings.dry_run}[/]\n")

    count = 0
    for msg, lead in fetch_hubspot_leads(settings, limit=limit):
        count += 1
        rprint(f"[cyan][{count}][/] Processing: {lead.first_name} {lead.last_name} <{lead.email}>")

        result = process_and_post(
            settings,
            lead,
            channel_id=target_channel,
            thread_ts=None,  # Post to main channel, not as thread
            enrich=enrich,
            max_searches=max_searches,
            include_lead_info=True,  # Include lead details in test posts
        )

        label_emoji = {"spam": "🔴", "solicitation": "🟡", "promising": "🟢"}.get(result.label, "⚪")
        rprint(f"    {label_emoji} {result.label.upper()} ({result.classification.confidence:.0%})")

        if settings.dry_run:
            rprint("    [dim](dry run - not posted)[/]")
        else:
            rprint(f"    [green]Posted to {target_channel}[/]")

    if count == 0:
        rprint("[yellow]No HubSpot leads found in channel history.[/]")
    else:
        rprint(f"\n[green]Processed {count} leads.[/]")


@app.command()
def replay(
    limit: int = typer.Option(5, "--limit", "-n", help="Number of leads to process"),
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Research promising leads with web search"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches per lead"),
    dry_run: bool = typer.Option(None, "--dry-run/--live", help="Override DRY_RUN config setting"),
    skip_replied: bool = typer.Option(True, "--skip-replied/--no-skip-replied", help="Skip already-replied leads"),
):
    """
    Replay mode: process historical leads and post as thread replies.

    Like production mode, but manually triggered on historical messages.
    Posts classification results as thread replies on the ORIGINAL messages
    in SLACK_CHANNEL_ID.

    Respects DRY_RUN config setting. Use --dry-run or --live to override.
    """
    from leads_agent.backtest import fetch_hubspot_leads
    from leads_agent.processor import process_and_post
    from leads_agent.slack import slack_client

    settings = get_settings()

    # Override dry_run if explicitly set
    if dry_run is not None:
        settings.dry_run = dry_run

    rprint(Panel.fit("🔄 [bold yellow]Replay Mode[/]", border_style="yellow"))
    rprint(f"[dim]Channel: {settings.slack_channel_id}[/]")
    rprint(f"[dim]Limit: {limit} | Enrich: {enrich} | Dry run: {settings.dry_run} | Skip replied: {skip_replied}[/]\n")

    if not settings.dry_run:
        if not Confirm.ask("[yellow]This will post replies to the production channel. Continue?[/]"):
            raise typer.Abort()

    client = slack_client(settings) if skip_replied else None

    count = 0
    skipped = 0
    for msg, lead in fetch_hubspot_leads(settings, limit=limit * 2 if skip_replied else limit):
        # Check if message already has replies
        if skip_replied and client:
            try:
                replies = client.conversations_replies(
                    channel=settings.slack_channel_id,
                    ts=msg["ts"],
                    limit=2,  # Just need to know if there are any replies
                )
                reply_count = len(replies.get("messages", [])) - 1  # Subtract the parent message
                if reply_count > 0:
                    skipped += 1
                    continue
            except Exception:
                pass  # If we can't check, process anyway

        if count >= limit:
            break

        count += 1
        rprint(f"[cyan][{count}][/] Processing: {lead.first_name} {lead.last_name} <{lead.email}>")
        rprint(f"    [dim]Message ts: {msg['ts']}[/]")

        result = process_and_post(
            settings,
            lead,
            channel_id=settings.slack_channel_id,
            thread_ts=msg["ts"],  # Reply to original message
            enrich=enrich,
            max_searches=max_searches,
            include_lead_info=False,  # Don't include lead info, it's in the parent message
        )

        label_emoji = {"spam": "🔴", "solicitation": "🟡", "promising": "🟢"}.get(result.label, "⚪")
        rprint(f"    {label_emoji} {result.label.upper()} ({result.classification.confidence:.0%})")

        if settings.dry_run:
            rprint("    [dim](dry run - not posted)[/]")
        else:
            rprint("    [green]Posted as thread reply[/]")

    if count == 0 and skipped == 0:
        rprint("[yellow]No HubSpot leads found in channel history.[/]")
    else:
        rprint(f"\n[green]Processed {count} leads.[/]")
        if skipped > 0:
            rprint(f"[dim]Skipped {skipped} leads that already had replies.[/]")


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
    enrich: bool = typer.Option(False, "--enrich", "-e", help="Research promising leads with web search"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches for enrichment"),
):
    """Classify a single message (for quick testing)."""
    from leads_agent.llm import ClassificationResult, classify_message
    from leads_agent.models import EnrichedLeadClassification

    settings = get_settings()

    title = "🧠 [bold yellow]Classifying Message[/]"
    if enrich:
        title += " [dim](with enrichment)[/]"
    rprint(Panel.fit(title, border_style="yellow"))
    rprint(f"[dim]{message}[/]\n")

    result = classify_message(settings, message, debug=debug, enrich=enrich, max_searches=max_searches)

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

    # Show enrichment results if available
    if isinstance(classification, EnrichedLeadClassification):
        if classification.company_research:
            rprint("\n[bold green]─── Company Research ───[/]")
            cr = classification.company_research
            rprint(f"[cyan]Company:[/] {cr.company_name}")
            rprint(f"[cyan]Description:[/] {cr.company_description}")
            if cr.industry:
                rprint(f"[cyan]Industry:[/] {cr.industry}")
            if cr.company_size:
                rprint(f"[cyan]Size:[/] {cr.company_size}")
            if cr.website:
                rprint(f"[cyan]Website:[/] {cr.website}")
            if cr.relevance_notes:
                rprint(f"[cyan]Relevance:[/] {cr.relevance_notes}")

        if classification.contact_research:
            rprint("\n[bold green]─── Contact Research ───[/]")
            cr = classification.contact_research
            rprint(f"[cyan]Name:[/] {cr.full_name}")
            if cr.title:
                rprint(f"[cyan]Title:[/] {cr.title}")
            if cr.linkedin_summary:
                rprint(f"[cyan]Summary:[/] {cr.linkedin_summary}")
            if cr.relevance_notes:
                rprint(f"[cyan]Relevance:[/] {cr.relevance_notes}")

        if classification.research_summary:
            rprint("\n[bold green]─── Research Summary ───[/]")
            rprint(classification.research_summary)

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
