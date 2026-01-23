"""Rich CLI for leads-agent."""

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from leads_agent.config import get_settings
from leads_agent.prompts import get_prompt_manager

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
    rprint("[dim]Create a Slack App at https://api.slack.com/apps[/]")
    rprint("[dim]Enable Socket Mode and generate an App-Level Token with connections:write scope[/]\n")

    slack_bot_token = Prompt.ask(
        "  [cyan]SLACK_BOT_TOKEN[/] [dim](xoxb-...)[/]",
        default="xoxb-...",
    )
    slack_app_token = Prompt.ask(
        "  [cyan]SLACK_APP_TOKEN[/] [dim](xapp-... for Socket Mode)[/]",
        default="xapp-...",
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

    rprint("\n[bold]Observability (Logfire)[/]")
    rprint("[dim]Get your token at https://logfire.pydantic.dev/[/]\n")
    logfire_token = Prompt.ask(
        "  [cyan]LOGFIRE_TOKEN[/] [dim](optional, for tracing)[/]",
        default="",
    )

    # Prompt Configuration
    rprint("\n[bold]Prompt Configuration[/] [dim](customize lead classification)[/]")
    configure_prompts = Confirm.ask("  Configure ICP and qualifying criteria?", default=False)

    prompt_config: dict = {}
    if configure_prompts:
        rprint("\n  [dim]Leave blank to skip any field[/]\n")

        company_name = Prompt.ask("  [cyan]Company name[/]", default="")
        if company_name:
            prompt_config["company_name"] = company_name

        services_desc = Prompt.ask("  [cyan]Services description[/]", default="")
        if services_desc:
            prompt_config["services_description"] = services_desc

        rprint("\n  [bold]Ideal Client Profile (ICP)[/]")
        icp_desc = Prompt.ask("  [cyan]ICP description[/] [dim](e.g., 'Mid-market B2B SaaS')[/]", default="")
        target_industries = Prompt.ask("  [cyan]Target industries[/] [dim](comma-separated)[/]", default="")
        target_sizes = Prompt.ask(
            "  [cyan]Target company sizes[/] [dim](e.g., SMB, Mid-Market, Enterprise)[/]", default=""
        )

        if any([icp_desc, target_industries, target_sizes]):
            icp: dict = {}
            if icp_desc:
                icp["description"] = icp_desc
            if target_industries:
                icp["target_industries"] = [s.strip() for s in target_industries.split(",")]
            if target_sizes:
                icp["target_company_sizes"] = [s.strip() for s in target_sizes.split(",")]
            prompt_config["icp"] = icp

        rprint("\n  [bold]Qualifying Questions[/] [dim](one per line, empty line to finish)[/]")
        questions = []
        while True:
            q = Prompt.ask("  [cyan]Question[/]", default="")
            if not q:
                break
            questions.append(q)
        if questions:
            prompt_config["qualifying_questions"] = questions

    # Build env content
    env_lines = [
        "# Slack credentials (Socket Mode)",
        f"SLACK_BOT_TOKEN={slack_bot_token}",
        f"SLACK_APP_TOKEN={slack_app_token}",
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

    # Logfire configuration
    if logfire_token:
        env_lines.extend(
            [
                "",
                "# Observability (Logfire)",
                f"LOGFIRE_TOKEN={logfire_token}",
            ]
        )
    else:
        env_lines.extend(
            [
                "",
                "# Observability (Logfire)",
                "# Get your token at https://logfire.pydantic.dev/",
                "# LOGFIRE_TOKEN=",
            ]
        )

    # Determine prompt config file path (same directory as .env)
    prompt_config_path = output.parent / "prompt_config.json"

    # Add prompt configuration reference
    if prompt_config:
        env_lines.extend(
            [
                "",
                "# Prompt Configuration (ICP, qualifying questions, etc.)",
                "# Points to JSON file - edit prompt_config.json to customize",
                f"PROMPT_CONFIG_PATH={prompt_config_path}",
            ]
        )
    else:
        env_lines.extend(
            [
                "",
                "# Prompt Configuration (ICP, qualifying questions, etc.)",
                "# Uncomment and create prompt_config.json to customize classification",
                "# See prompt_config.example.json for all available options",
                f"# PROMPT_CONFIG_PATH={prompt_config_path}",
            ]
        )

    env_content = "\n".join(env_lines) + "\n"

    # Write .env file
    output.write_text(env_content)
    rprint(f"\n[green]✓[/] Configuration written to [bold]{output}[/]")

    # Write prompt_config.json if configured
    if prompt_config:
        prompt_config_content = json.dumps(prompt_config, indent=2)
        prompt_config_path.write_text(prompt_config_content + "\n")
        rprint(f"[green]✓[/] Prompt configuration written to [bold]{prompt_config_path}[/]")
    else:
        rprint(f"[dim]To customize prompts, create {prompt_config_path} (see prompt_config.example.json)[/]")

    rprint("\n[dim]Run [bold]leads-agent config[/] to verify settings[/]")
    rprint("[dim]Run [bold]leads-agent prompts[/] to view prompt configuration[/]")


def _mask(secret, visible: int = 4) -> str:
    """Mask a secret string, handling SecretStr or None."""
    if secret is None:
        return "[not set]"
    # Handle pydantic SecretStr
    val = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
    if len(val) <= visible:
        return "***"
    return val[:visible] + "*" * (len(val) - visible)


def _find_prompt_config_source() -> str | None:
    """Find where prompt configuration is being loaded from."""
    import os

    # Check env var first
    env_path = os.environ.get("PROMPT_CONFIG_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    # Check default locations
    candidates = [
        Path("prompt_config.json"),
        Path("config/prompt_config.json"),
        Path.cwd() / "prompt_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


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
    table.add_row("SLACK_APP_TOKEN", _mask(settings.slack_app_token))
    table.add_row("SLACK_CHANNEL_ID", settings.slack_channel_id or "[not set]")
    table.add_row("SLACK_TEST_CHANNEL_ID", settings.slack_test_channel_id or "[not set]")
    table.add_row("OPENAI_API_KEY", _mask(settings.openai_api_key))
    table.add_row("LLM_BASE_URL", settings.llm_base_url)
    table.add_row("LLM_MODEL_NAME", settings.llm_model_name)
    table.add_row("DRY_RUN", str(settings.dry_run))

    # Show prompt config path
    prompt_config_source = _find_prompt_config_source()
    table.add_row("PROMPT_CONFIG", prompt_config_source or "[default]")

    console.print(table)

    if prompt_config_source:
        rprint("\n[dim]Run [bold]leads-agent prompts[/] to view prompt configuration[/]")


@app.command()
def prompts(
    show_full: bool = typer.Option(False, "--full", "-f", help="Show full rendered prompts"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output configuration as JSON"),
):
    """Display current prompt configuration."""
    manager = get_prompt_manager()
    config = manager.config

    if as_json:
        # Output raw JSON for scripting
        rprint(json.dumps(config.model_dump(exclude_none=True), indent=2))
        return

    rprint(Panel.fit("📝 [bold cyan]Prompt Configuration[/]", border_style="cyan"))

    # Show source of configuration
    config_source = _find_prompt_config_source()
    if config_source:
        rprint(f"[dim]Loaded from: {config_source}[/]\n")

    # Check if configuration is empty
    if config.is_empty():
        rprint("[yellow]No custom prompt configuration set.[/]")
        rprint("[dim]Using default prompts. To customize:[/]")
        rprint("[dim]  1. Run [bold]leads-agent init[/] and configure prompts[/]")
        rprint("[dim]  2. Set PROMPT_CONFIG_JSON environment variable[/]")
        rprint("[dim]  3. Create prompt_config.json file (see prompt_config.example.json)[/]")
        rprint("[dim]  4. Use the API: PUT /config/prompts[/]")

        if show_full:
            rprint("\n[bold]Default Classification Prompt:[/]")
            rprint(Syntax(manager.build_classification_prompt(), "text", theme="monokai", word_wrap=True))
        return

    # Show company info
    if config.company_name or config.services_description:
        rprint("[bold]Company:[/]")
        if config.company_name:
            rprint(f"  [cyan]Name:[/] {config.company_name}")
        if config.services_description:
            rprint(f"  [cyan]Services:[/] {config.services_description}")
        rprint()

    # ICP section
    if config.icp:
        icp = config.icp
        rprint("[bold]Ideal Client Profile (ICP):[/]")
        if icp.description:
            rprint(f"  [cyan]Description:[/] {icp.description}")
        if icp.target_industries:
            rprint(f"  [cyan]Industries:[/] {', '.join(icp.target_industries)}")
        if icp.target_company_sizes:
            rprint(f"  [cyan]Company Sizes:[/] {', '.join(icp.target_company_sizes)}")
        if icp.target_roles:
            rprint(f"  [cyan]Target Roles:[/] {', '.join(icp.target_roles)}")
        if icp.geographic_focus:
            rprint(f"  [cyan]Geographic Focus:[/] {', '.join(icp.geographic_focus)}")
        if icp.disqualifying_signals:
            rprint(f"  [cyan]Disqualifying:[/] {', '.join(icp.disqualifying_signals)}")

    # Qualifying questions
    if config.qualifying_questions:
        rprint("\n[bold]Qualifying Questions:[/]")
        for i, q in enumerate(config.qualifying_questions, 1):
            rprint(f"  [dim]{i}.[/] {q}")

    # Custom instructions
    if config.custom_instructions:
        rprint("\n[bold]Custom Instructions:[/]")
        rprint(f"  [dim]{config.custom_instructions}[/]")

    # Research focus areas
    if config.research_focus_areas:
        rprint("\n[bold]Research Focus Areas:[/]")
        for area in config.research_focus_areas:
            rprint(f"  • {area}")

    # Show full prompts if requested
    if show_full:
        rprint("\n" + "─" * 60)
        rprint("[bold]Full Classification Prompt:[/]")
        rprint(Syntax(manager.build_classification_prompt(), "text", theme="monokai", word_wrap=True))

        rprint("\n" + "─" * 60)
        rprint("[bold]Full Research Prompt:[/]")
        rprint(Syntax(manager.build_research_prompt(), "text", theme="monokai", word_wrap=True))


@app.command()
def run():
    """Start the Slack bot using Socket Mode."""
    from leads_agent.bolt_app import run_socket_mode

    rprint(Panel.fit("🔌 [bold green]Starting Leads Agent (Socket Mode)[/]", border_style="green"))
    run_socket_mode()


@app.command()
def collect(
    keep: int = typer.Option(20, "--keep", "-n", help="Number of events to collect"),
    output: str = typer.Option("collected_events.json", "--output", "-o", help="Output JSON file"),
):
    """
    Collect raw Socket Mode events for debugging.

    Saves raw event payloads exactly as received from Slack.
    Useful for inspecting event format and structure.
    """
    from leads_agent.bolt_app import collect_events

    rprint(Panel.fit("📡 [bold blue]Collecting Raw Socket Mode Events[/]", border_style="blue"))
    collect_events(keep=keep, output_file=output)


@app.command()
def backtest(
    events_file: Path = typer.Argument(..., help="JSON file with collected events (from `collect` command)"),
    limit: int = typer.Option(None, "--limit", "-n", help="Max number of leads to process"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches per lead"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show agent steps and token usage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full message history (with --debug)"),
):
    """
    Run classifier on collected events (console output only).

    First collect events with: leads-agent collect --keep 20
    Then backtest with: leads-agent backtest collected_events.json
    """
    from leads_agent.backtest import run_backtest

    modes = []
    if debug:
        modes.append("debug")
    mode_str = f" [dim]({', '.join(modes)})[/]" if modes else ""
    title = f"🔬 [bold magenta]Backtesting Lead Classifier[/]{mode_str}"
    rprint(Panel.fit(title, border_style="magenta"))
    rprint(f"[dim]Events file: {events_file}[/]\n")
    run_backtest(events_file=events_file, limit=limit, max_searches=max_searches, debug=debug, verbose=verbose)


@app.command()
def test(
    test_channel: str = typer.Option(None, "--channel", "-c", help="Test channel ID to post to"),
    dry_run: bool = typer.Option(None, "--dry-run/--live", help="Override DRY_RUN config setting"),
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches per lead"),
):
    """
    Test mode: listen via Socket Mode, post results to test channel.

    Like production mode, but posts to SLACK_TEST_CHANNEL_ID instead of
    replying in threads. Useful for testing the full pipeline.

    Respects DRY_RUN config setting. Use --dry-run or --live to override.
    """
    from leads_agent.bolt_app import run_test_mode

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

    rprint(Panel.fit("🧪 [bold cyan]Test Mode (Socket Mode)[/]", border_style="cyan"))
    rprint(f"[dim]Listening for HubSpot messages → Posting to {target_channel}[/]")
    rprint(f"[dim]Dry run: {settings.dry_run}[/]\n")

    run_test_mode(settings=settings, test_channel=target_channel, max_searches=max_searches)


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
    max_searches: int = typer.Option(4, "--max-searches", help="Max web searches for enrichment"),
):
    """Classify a single message (for quick testing)."""
    from leads_agent.agent import ClassificationResult, classify_message
    from leads_agent.models import EnrichedLeadClassification

    settings = get_settings()

    title = "🧠 [bold yellow]Classifying Message[/]"
    rprint(Panel.fit(title, border_style="yellow"))
    rprint(f"[dim]{message}[/]\n")

    result = classify_message(settings, message, debug=debug, max_searches=max_searches)

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

    decision_color = {"ignore": "red", "promising": "green"}.get(label_value, "white")
    table.add_row("Decision", f"[bold {decision_color}]{label_value}[/]")
    table.add_row("Confidence", f"{confidence:.0%}")
    table.add_row("Reason", reason)

    if getattr(classification, "score", None) is not None:
        table.add_row("Score", f"{classification.score}/5")
    if getattr(classification, "action", None) is not None:
        table.add_row("Action", classification.action.value)
    if getattr(classification, "score_reason", None):
        table.add_row("Score Reason", classification.score_reason)

    if classification.lead_summary:
        table.add_row("Summary", classification.lead_summary)
    if classification.key_signals:
        table.add_row("Signals", ", ".join(classification.key_signals))

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
