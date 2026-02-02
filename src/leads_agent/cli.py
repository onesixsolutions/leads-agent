from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from leads_agent.config import get_settings

app = typer.Typer(
    name="leads-agent",
    help="🧠 AI-powered Slack lead classifier",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


@app.command(name="init")
def init_command(
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
    from leads_agent.core import init_wizard

    init_wizard(output, force)


@app.command(name="config")
def config_command():
    """Display current configuration (from environment)."""
    from leads_agent.config import display_config

    display_config()


@app.command(name="prompts")
def prompts_command(
    show_full: bool = typer.Option(
        False, "--full", "-f", help="Show full rendered prompts"
    ),
    as_json: bool = typer.Option(
        False, "--json", "-j", help="Output configuration as JSON"
    ),
):
    """Display current prompt configuration."""
    from leads_agent.prompts import display_prompts

    display_prompts(show_full, as_json)


@app.command(name="run")
def run_command():
    """Start the Slack bot using Socket Mode."""
    from leads_agent.app import run_socket_mode

    rprint(Panel.fit("🔌 [bold green]Starting Leads Agent[/]", border_style="green"))
    run_socket_mode()


@app.command()
def collect(
    keep: int = typer.Option(20, "--keep", "-n", help="Number of events to collect"),
    output: str = typer.Option(
        "collected_events.json", "--output", "-o", help="Output JSON file"
    ),
):
    """
    Collect raw Socket Mode events for debugging.

    Saves raw event payloads exactly as received from Slack.
    Useful for inspecting event format and structure.
    """
    from leads_agent.app import collect_events

    rprint(
        Panel.fit(
            "📡 [bold blue]Collecting Raw Socket Mode Events[/]", border_style="blue"
        )
    )
    collect_events(keep=keep, output_file=output)


@app.command(name="backtest")
def backtest_command(
    events_file: Path = typer.Argument(
        ..., help="JSON file with collected events (from `collect` command)"
    ),
    limit: int = typer.Option(
        None, "--limit", "-n", help="Max number of leads to process"
    ),
    max_searches: int = typer.Option(
        4, "--max-searches", help="Max web searches per lead"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show agent steps and token usage"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full message history (with --debug)"
    ),
):
    """
    Run classifier on collected events (console output only).

    First collect events with: leads-agent collect --keep 20
    Then backtest with: leads-agent backtest collected_events.json
    """
    from leads_agent.core import run_backtest

    modes = []
    if debug:
        modes.append("debug")
    mode_str = f" [dim]({', '.join(modes)})[/]" if modes else ""
    title = f"🔬 [bold magenta]Backtesting Lead Classifier[/]{mode_str}"
    rprint(Panel.fit(title, border_style="magenta"))
    rprint(f"[dim]Events file: {events_file}[/]\n")
    run_backtest(
        events_file=events_file,
        limit=limit,
        max_searches=max_searches,
        debug=debug,
        verbose=verbose,
    )


@app.command()
def test(
    test_channel: str = typer.Option(
        None, "--channel", "-c", help="Test channel ID to post to"
    ),
    dry_run: bool = typer.Option(
        None, "--dry-run/--live", help="Override DRY_RUN config setting"
    ),
    max_searches: int = typer.Option(
        4, "--max-searches", help="Max web searches per lead"
    ),
):
    """
    Test mode: listen via Socket Mode, post results to test channel.

    Like production mode, but posts to SLACK_TEST_CHANNEL_ID instead of
    replying in threads. Useful for testing the full pipeline.

    Respects DRY_RUN config setting. Use --dry-run or --live to override.
    """
    from leads_agent.app import run_test_mode

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

    run_test_mode(
        settings=settings, test_channel=target_channel, max_searches=max_searches
    )


@app.command(name="pull-history")
def pull_history_command(
    output: Path = typer.Option(
        Path("channel_history.json"),
        "--output",
        "-o",
        help="Output JSON file path",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Number of messages to fetch"),
    channel_id: str = typer.Option(
        None, "--channel", "-c", help="Channel ID (defaults to SLACK_CHANNEL_ID)"
    ),
    print_only: bool = typer.Option(
        False, "--print", "-p", help="Print to console instead of saving to file"
    ),
):
    """Fetch Slack channel history and save to JSON."""
    from leads_agent.core import pull_history

    pull_history(
        channel_id=channel_id, limit=limit, output=output, print_only=print_only
    )


@app.command(name="replay")
def replay_command(
    limit: int = typer.Option(
        1, "--limit", "-n", help="Number of HubSpot lead messages to replay"
    ),
    channel_id: str = typer.Option(
        None, "--channel", "-c", help="Channel ID (defaults to SLACK_CHANNEL_ID)"
    ),
    dry_run: bool = typer.Option(
        None, "--dry-run/--live", help="Dry-run prints output instead of posting"
    ),
    max_searches: int = typer.Option(
        4, "--max-searches", help="Max web searches per lead"
    ),
):
    """
    Replay HubSpot lead messages from Slack channel history.

    Fetches recent channel messages, finds HubSpot bot lead messages, runs the normal
    processing pipeline, and either posts the result back to Slack (thread reply) or
    prints the generated message when --dry-run is enabled.
    """
    from leads_agent.core import replay

    replay(
        channel_id=channel_id, limit=limit, dry_run=dry_run, max_searches=max_searches
    )


@app.command(name="classify")
def classify_command(
    message: str = typer.Argument(..., help="Message text to classify"),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show message history and agent trace"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full message content (no truncation)"
    ),
    max_searches: int = typer.Option(
        4, "--max-searches", help="Max web searches for enrichment"
    ),
):
    """Classify a single message (for quick testing)."""
    from leads_agent.core import classify

    classify(message, debug, max_searches, verbose)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
