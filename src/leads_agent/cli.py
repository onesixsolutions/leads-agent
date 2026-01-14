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
    rprint("[dim]Default uses local Ollama; change for OpenAI/other providers[/]\n")

    llm_base_url = Prompt.ask(
        "  [cyan]LLM_BASE_URL[/]",
        default="http://localhost:11434/v1",
    )
    llm_model_name = Prompt.ask(
        "  [cyan]LLM_MODEL_NAME[/]",
        default="llama3.1:8b",
    )

    rprint("\n[bold]Runtime Options[/]")
    dry_run = Confirm.ask("  [cyan]DRY_RUN[/] (don't post replies)?", default=True)

    env_content = f"""\
# Slack credentials
SLACK_BOT_TOKEN={slack_bot_token}
SLACK_SIGNING_SECRET={slack_signing_secret}
SLACK_CHANNEL_ID={slack_channel_id}

# LLM configuration
LLM_BASE_URL={llm_base_url}
LLM_MODEL_NAME={llm_model_name}

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
    table.add_row("SLACK_CHANNEL_ID", settings.slack_channel_id)
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


@app.command()
def classify(
    message: str = typer.Argument(..., help="Message text to classify"),
):
    """Classify a single message (for quick testing)."""
    from leads_agent.llm import classify_message

    settings = get_settings()

    rprint(Panel.fit("🧠 [bold yellow]Classifying Message[/]", border_style="yellow"))
    rprint(f"[dim]{message}[/]\n")

    result = classify_message(settings, message)

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    label_color = {"spam": "red", "solicitation": "yellow", "promising": "green"}.get(result.label.value, "white")

    table.add_row("Label", f"[bold {label_color}]{result.label.value}[/]")
    table.add_row("Confidence", f"{result.confidence:.0%}")
    table.add_row("Reason", result.reason)

    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
