from pathlib import Path
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
import typer
import json

def init_wizard(output: Path, force: bool):

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
    debug = Confirm.ask("  [cyan]DEBUG[/] (log incoming events)?", default=True)

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
            f"DEBUG={str(debug).lower()}",
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