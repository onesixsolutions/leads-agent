"""
Console rendering of an ICP assessment.

Shared by `backtest` and `classify` so the two views cannot drift. Slack gets
a short card (`core.processor.format_slack_message`) plus a threaded full
brief; the hosted HTML version lives in `leads_agent.briefs`.
"""

from __future__ import annotations

from leads_agent.icp_fit import CRITERIA, verdict_display
from leads_agent.models import CriterionStatus, ICPVerdict

_STATUS_MARK: dict[CriterionStatus, str] = {
    CriterionStatus.met: "[green]✓ met     [/]",
    CriterionStatus.not_met: "[red]✗ not met [/]",
    CriterionStatus.partial: "[yellow]~ partial [/]",
    CriterionStatus.unknown: "[dim]? unknown [/]",
}


def format_icp_report(classification: object) -> str:
    """
    Render the ICP verdict, criteria table, reasons and brief as rich markup.

    Returns an empty string when the lead has no assessment (e.g. spam), so
    callers can print unconditionally.
    """
    verdict: ICPVerdict | None = getattr(classification, "icp_verdict", None)
    assessment = getattr(classification, "icp_assessment", None)

    if verdict is None or verdict == ICPVerdict.not_evaluated or assessment is None:
        return ""

    emoji, label = verdict_display(verdict)
    lines: list[str] = ["", f"{emoji} [bold]ICP VERDICT: {label}[/]"]

    brief = getattr(classification, "brief", None)
    if brief and brief.icp_statement:
        lines.append(f"[italic]{brief.icp_statement}[/]")

    # Criteria table — every lead reports the same rows, in the same order.
    lines.append("")
    lines.append("[bold]Criteria[/]")
    for spec in CRITERIA:
        criterion = getattr(assessment, spec.field, None)
        if criterion is None:
            continue
        mark = _STATUS_MARK.get(criterion.status, "[dim]?[/]")
        gate = "[dim](gate)[/]" if spec.is_gate else ""
        lines.append(f"  {mark} [cyan]{spec.label}[/] {gate}")
        if criterion.finding:
            lines.append(f"        {criterion.finding}")
        if criterion.evidence:
            lines.append(f"        [dim]evidence: {criterion.evidence}[/]")

    for title, style, items in (
        ("Why NOT in ICP", "red", getattr(classification, "reasons_out_of_icp", None)),
        ("Why in ICP", "green", getattr(classification, "reasons_in_icp", None)),
        ("Unverified — check before pursuing", "yellow", getattr(classification, "open_questions", None)),
    ):
        if items:
            lines.append("")
            lines.append(f"[bold {style}]{title}[/]")
            lines.extend(f"  • {item}" for item in items)

    if brief:
        if brief.analyst_take:
            lines.extend(["", f"[bold]🧠 Take[/] {brief.analyst_take}"])
        if brief.opportunity:
            lines.append(f"[bold green]📈 Opportunity[/] {brief.opportunity}")
        if brief.risks:
            lines.append("[bold yellow]⚠️  Risks[/] " + "; ".join(brief.risks))
        if brief.exception_case:
            lines.append(f"[bold magenta]🚩 Exception case[/] {brief.exception_case}")
        if brief.recommended_entry:
            lines.append(f"[bold]🚪 Recommended entry[/] {brief.recommended_entry}")
        if brief.accelerators:
            lines.append("[bold]🧰 Accelerators[/] " + ", ".join(brief.accelerators))
        if brief.talking_points:
            lines.append("[bold]💬 Talking points[/]")
            lines.extend(f"  • {t}" for t in brief.talking_points)

    action = getattr(classification, "action", None)
    if action is not None:
        lines.extend(["", f"[bold]Action:[/] {action.value}"])

    return "\n".join(lines)
