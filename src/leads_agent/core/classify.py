from rich import print as rprint
from rich.console import Console
from rich.table import Table

from leads_agent.agent import ClassificationResult, classify_message
from leads_agent.config import get_settings
from leads_agent.icp_fit import verdict_display
from leads_agent.core.icp_report import format_icp_report
from leads_agent.models import EnrichedLeadClassification

console = Console()


def classify(message: str, debug: bool, max_searches: int, verbose: bool):
    settings = get_settings()

    result = classify_message(settings, message, debug=debug, max_searches=max_searches)

    # Handle both return types
    if isinstance(result, ClassificationResult):
        classification = result.classification
        label_value = result.label
        reason = result.reason
    else:
        classification = result
        label_value = result.label.value
        reason = result.reason

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    decision_color = {"ignore": "red", "promising": "green"}.get(label_value, "white")
    table.add_row("Decision", f"[bold {decision_color}]{label_value}[/]")
    table.add_row("Reason", reason)

    verdict = getattr(classification, "icp_verdict", None)
    if verdict is not None:
        emoji, vlabel = verdict_display(verdict)
        table.add_row("ICP Verdict", f"[bold]{emoji} {vlabel}[/]")
    if getattr(classification, "action", None) is not None:
        table.add_row("Action", classification.action.value)

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

    report = format_icp_report(classification)
    if report.strip():
        rprint(report)

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
