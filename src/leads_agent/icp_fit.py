"""
Deterministic ICP verdict derivation.

The LLM's job is to establish *findings with evidence* for each ICP criterion.
Deciding what those findings add up to is this module's job, in plain Python,
so that the same evidence always produces the same verdict and every
disqualification names the rule that caused it.

Rule precedence:

1. Any GATE criterion `not_met`            -> out_of_icp
2. All gates `met`/`partial`, none unknown -> in_icp
3. Any gate `unknown`                      -> needs_verification
4. Otherwise                               -> partial_fit
"""

from __future__ import annotations

from dataclasses import dataclass

from leads_agent.models import (
    CriterionStatus,
    EnrichedLeadClassification,
    ICPAssessment,
    ICPVerdict,
    LeadAction,
)


@dataclass(frozen=True)
class CriterionSpec:
    """How one criterion is labelled and whether it can disqualify a lead."""

    field: str
    label: str
    # A gate can disqualify outright when `not_met`. Non-gates are reported
    # but never change the verdict.
    is_gate: bool
    # Some gates are hard ICP rules (band, floor, platform, door). Others are
    # softer: missing them caps the lead rather than killing it.
    hard: bool = False


# Order here is the order shown to the user: hard gates first, then softer
# gates, then informational context.
CRITERIA: tuple[CriterionSpec, ...] = (
    # A company we cannot find at all is a red flag in its own right: a
    # $250M-$10B business has a web presence. Note this judges the COMPANY,
    # never the individual — private mid-market executives are routinely hard
    # to find, which is why contact findability is not a criterion at all.
    CriterionSpec("company_footprint", "Company identifiable in search", is_gate=True, hard=True),
    CriterionSpec("revenue_band", "Revenue band ($250M–$10B)", is_gate=True, hard=True),
    CriterionSpec("deal_shape", "Deal shape ($150k+ floor)", is_gate=True, hard=True),
    CriterionSpec("platform", "Platform trajectory", is_gate=True, hard=True),
    CriterionSpec("entry_door", "Entry door", is_gate=True, hard=True),
    CriterionSpec("executive_sponsor", "Executive sponsor & budget", is_gate=True),
    CriterionSpec("internal_team_depth", "Thin internal data/ML team", is_gate=True),
    CriterionSpec("trigger", "Trigger / urgency", is_gate=False),
    CriterionSpec("expansion_path", "Designed expansion path", is_gate=False),
    CriterionSpec("buyer_persona", "Buyer persona fit", is_gate=False),
    CriterionSpec("focus_overlay", "Focus overlay (informational)", is_gate=False),
)

GATE_FIELDS = tuple(c.field for c in CRITERIA if c.is_gate)


def _spec(field: str) -> CriterionSpec:
    return next(c for c in CRITERIA if c.field == field)


def derive_verdict(assessment: ICPAssessment) -> tuple[ICPVerdict, list[str], list[str], list[str]]:
    """
    Derive the overall verdict plus the reasons for, against, and outstanding.

    Returns:
        (verdict, reasons_in_icp, reasons_out_of_icp, open_questions)
    """
    reasons_in: list[str] = []
    reasons_out: list[str] = []
    open_questions: list[str] = []

    failed_gates: list[str] = []
    unknown_gates: list[str] = []

    for spec in CRITERIA:
        criterion = getattr(assessment, spec.field)
        finding = (criterion.finding or "").strip()
        line = f"{spec.label}: {finding}" if finding else spec.label

        if criterion.status == CriterionStatus.not_met:
            if spec.is_gate:
                failed_gates.append(spec.field)
                reasons_out.append(line)
            else:
                # Non-gate misses are context, not disqualifiers.
                reasons_out.append(f"{line} (not disqualifying)")
        elif criterion.status == CriterionStatus.met:
            reasons_in.append(line)
        elif criterion.status == CriterionStatus.partial:
            reasons_in.append(f"{line} (partial)")
        else:  # unknown
            if spec.is_gate:
                unknown_gates.append(spec.field)
            open_questions.append(f"{spec.label}: {finding or 'not established'}")

    if failed_gates:
        verdict = ICPVerdict.out_of_icp
    elif not unknown_gates:
        verdict = ICPVerdict.in_icp
    else:
        # Nothing disqualifying, but a gate is unproven. If the hard gates are
        # all established and only softer gates are open, the lead is a
        # partial fit rather than unverified.
        unknown_hard = [f for f in unknown_gates if _spec(f).hard]
        verdict = ICPVerdict.needs_verification if unknown_hard else ICPVerdict.partial_fit

    return verdict, reasons_in, reasons_out, open_questions


def action_for(verdict: ICPVerdict) -> LeadAction:
    """Map a verdict to the recommended action."""
    if verdict in (ICPVerdict.out_of_icp, ICPVerdict.not_evaluated):
        return LeadAction.ignore
    return LeadAction.follow_up


def apply_icp_fit(classification: EnrichedLeadClassification) -> EnrichedLeadClassification:
    """
    Populate the derived ICP fields on a classification, in place.

    Safe to call on any classification: a lead with no assessment (spam, or a
    research failure) resolves to `not_evaluated` rather than erroring.
    """
    if classification.icp_assessment is None:
        classification.icp_verdict = ICPVerdict.not_evaluated
        classification.reasons_in_icp = None
        classification.reasons_out_of_icp = None
        classification.open_questions = None
        classification.action = action_for(ICPVerdict.not_evaluated)
        return classification

    verdict, reasons_in, reasons_out, open_questions = derive_verdict(
        classification.icp_assessment
    )

    classification.icp_verdict = verdict
    classification.reasons_in_icp = reasons_in or None
    classification.reasons_out_of_icp = reasons_out or None
    classification.open_questions = open_questions or None
    classification.action = action_for(verdict)

    return classification


VERDICT_DISPLAY: dict[ICPVerdict, tuple[str, str]] = {
    ICPVerdict.in_icp: ("✅", "IN ICP"),
    ICPVerdict.partial_fit: ("🟡", "PARTIAL FIT"),
    ICPVerdict.needs_verification: ("🔍", "NEEDS VERIFICATION"),
    ICPVerdict.out_of_icp: ("⛔", "NOT IN ICP"),
    ICPVerdict.not_evaluated: ("🚫", "NOT EVALUATED"),
}


def verdict_display(verdict: ICPVerdict | None) -> tuple[str, str]:
    """Emoji + label for a verdict."""
    if verdict is None:
        return ("❓", "UNKNOWN")
    return VERDICT_DISPLAY.get(verdict, ("❓", verdict.value.upper()))
