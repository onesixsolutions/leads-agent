from rich import print as rprint
from rich.panel import Panel
from rich.syntax import Syntax
import json

from leads_agent.prompts.manager import get_prompt_manager
from leads_agent.config import _find_prompt_config_source

def display_prompts(show_full: bool = False, as_json: bool = False):
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