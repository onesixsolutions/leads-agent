"""
Tests for the deterministic ICP verdict logic.

These run offline — no LLM, no network. They pin the rules that must not drift:
`unknown` never disqualifies, hard gates do, and non-gates never do.
"""

from __future__ import annotations

import pytest

from leads_agent.icp_fit import CRITERIA, apply_icp_fit, derive_verdict
from leads_agent.models import (
    CriterionStatus,
    EnrichedLeadClassification,
    ICPAssessment,
    ICPCriterion,
    ICPVerdict,
    LeadAction,
    LeadLabel,
)

ALL_FIELDS = [c.field for c in CRITERIA]
GATES = [c.field for c in CRITERIA if c.is_gate]
HARD_GATES = [c.field for c in CRITERIA if c.is_gate and c.hard]
SOFT_GATES = [c.field for c in CRITERIA if c.is_gate and not c.hard]
NON_GATES = [c.field for c in CRITERIA if not c.is_gate]


def _criterion(status: CriterionStatus, finding: str = "finding") -> ICPCriterion:
    return ICPCriterion(
        status=status,
        finding=finding,
        evidence=None if status == CriterionStatus.unknown else "evidence",
    )


def build_assessment(default: CriterionStatus = CriterionStatus.met, **overrides) -> ICPAssessment:
    """Assessment with every criterion at `default`, then per-field overrides."""
    fields = {name: _criterion(default) for name in ALL_FIELDS}
    for name, status in overrides.items():
        fields[name] = _criterion(status, finding=f"{name} is {status.value}")
    return ICPAssessment(**fields)


# --- Verdict derivation ---------------------------------------------------


def test_all_met_is_in_icp():
    verdict, reasons_in, reasons_out, _ = derive_verdict(build_assessment())
    assert verdict == ICPVerdict.in_icp
    assert reasons_in and not reasons_out


@pytest.mark.parametrize("gate", HARD_GATES)
def test_any_hard_gate_not_met_is_out_of_icp(gate):
    assessment = build_assessment(**{gate: CriterionStatus.not_met})
    verdict, _, reasons_out, _ = derive_verdict(assessment)
    assert verdict == ICPVerdict.out_of_icp
    # The failing gate must be named in the reasons — a silent rejection is
    # exactly what this design exists to prevent.
    assert any(gate.replace("_", " ") in r.lower() or f"{gate} is not_met" in r for r in reasons_out)


@pytest.mark.parametrize("gate", SOFT_GATES)
def test_soft_gate_not_met_also_disqualifies(gate):
    """Soft gates are still gates: positive evidence against them disqualifies."""
    verdict, _, reasons_out, _ = derive_verdict(build_assessment(**{gate: CriterionStatus.not_met}))
    assert verdict == ICPVerdict.out_of_icp
    assert reasons_out


@pytest.mark.parametrize("field", NON_GATES)
def test_non_gate_not_met_never_disqualifies(field):
    """A missing trigger or an off-overlay industry must not reject a lead."""
    verdict, _, reasons_out, _ = derive_verdict(build_assessment(**{field: CriterionStatus.not_met}))
    assert verdict == ICPVerdict.in_icp
    assert any("not disqualifying" in r for r in reasons_out)


@pytest.mark.parametrize("gate", HARD_GATES)
def test_unknown_hard_gate_needs_verification_not_rejection(gate):
    """
    The most important rule in the module: absent evidence is not a rejection.
    """
    verdict, _, reasons_out, open_questions = derive_verdict(
        build_assessment(**{gate: CriterionStatus.unknown})
    )
    assert verdict == ICPVerdict.needs_verification
    assert not reasons_out
    assert open_questions


@pytest.mark.parametrize("gate", SOFT_GATES)
def test_unknown_soft_gate_is_partial_fit(gate):
    verdict, _, _, open_questions = derive_verdict(build_assessment(**{gate: CriterionStatus.unknown}))
    assert verdict == ICPVerdict.partial_fit
    assert open_questions


def test_partial_platform_still_in_icp():
    """A non-Snowflake modern platform qualifies with no penalty."""
    verdict, reasons_in, reasons_out, _ = derive_verdict(
        build_assessment(platform=CriterionStatus.partial)
    )
    assert verdict == ICPVerdict.in_icp
    assert not reasons_out
    assert any("partial" in r for r in reasons_in)


def test_failed_gate_beats_unknown_gate():
    """A hard failure is decisive even when other gates are unproven."""
    verdict, _, _, _ = derive_verdict(
        build_assessment(
            revenue_band=CriterionStatus.not_met,
            platform=CriterionStatus.unknown,
        )
    )
    assert verdict == ICPVerdict.out_of_icp


def test_all_unknown_needs_verification():
    verdict, reasons_in, reasons_out, open_questions = derive_verdict(
        build_assessment(default=CriterionStatus.unknown)
    )
    assert verdict == ICPVerdict.needs_verification
    assert not reasons_in and not reasons_out
    assert len(open_questions) == len(ALL_FIELDS)


# --- Action mapping ------------------------------------------------------


@pytest.mark.parametrize(
    ("assessment_kwargs", "expected_verdict", "expected_action"),
    [
        ({}, ICPVerdict.in_icp, LeadAction.follow_up),
        ({"revenue_band": CriterionStatus.not_met}, ICPVerdict.out_of_icp, LeadAction.ignore),
        ({"platform": CriterionStatus.unknown}, ICPVerdict.needs_verification, LeadAction.follow_up),
        ({"executive_sponsor": CriterionStatus.unknown}, ICPVerdict.partial_fit, LeadAction.follow_up),
    ],
)
def test_apply_icp_fit_sets_verdict_and_action(assessment_kwargs, expected_verdict, expected_action):
    classification = EnrichedLeadClassification(
        label=LeadLabel.promising,
        reason="genuine inquiry",
        icp_assessment=build_assessment(**assessment_kwargs),
    )
    result = apply_icp_fit(classification)
    assert result.icp_verdict == expected_verdict
    assert result.action == expected_action


def test_prioritize_action_no_longer_exists():
    """`prioritize` was removed: fit is expressed by the verdict, not the action."""
    assert {a.value for a in LeadAction} == {"ignore", "follow_up"}


def test_no_confidence_or_score_fields():
    """The arbitrary percentage and 1-5 score are gone from the contract."""
    fields = set(EnrichedLeadClassification.model_fields)
    assert not fields & {"confidence", "score", "score_reason"}


def test_unfindable_company_is_a_red_flag():
    """
    Deliberate asymmetry: for `company_footprint`, absence of evidence IS
    evidence. A $250M-$10B business has a web presence, so a company that
    cannot be found at all is out of ICP rather than merely unverified.
    """
    verdict, _, reasons_out, _ = derive_verdict(
        build_assessment(company_footprint=CriterionStatus.not_met)
    )
    assert verdict == ICPVerdict.out_of_icp
    assert reasons_out


def test_no_criterion_scores_contact_findability():
    """
    A hard-to-find individual is normal at privately held mid-market companies
    and must never count against a lead, so there is no criterion for it.
    """
    fields = set(ICPAssessment.model_fields)
    # No criterion may exist for locating the individual. (`buyer_persona` is
    # about which persona they are, not whether they can be found.)
    assert not any("contact" in f or "findab" in f for f in fields)
    # And the footprint criterion must be explicitly scoped to the company.
    desc = ICPAssessment.model_fields["company_footprint"].description
    assert "COMPANY" in desc and "individual" in desc


def test_missing_assessment_is_not_evaluated():
    """Spam and research failures must not crash the derivation."""
    classification = EnrichedLeadClassification(label=LeadLabel.ignore, reason="spam")
    result = apply_icp_fit(classification)
    assert result.icp_verdict == ICPVerdict.not_evaluated
    assert result.action == LeadAction.ignore
    assert result.reasons_out_of_icp is None


def test_model_supplied_verdict_is_overwritten():
    """
    The model must not be able to choose its own verdict — if it sets one, the
    derivation replaces it.
    """
    classification = EnrichedLeadClassification(
        label=LeadLabel.promising,
        reason="genuine inquiry",
        icp_assessment=build_assessment(revenue_band=CriterionStatus.not_met),
        icp_verdict=ICPVerdict.in_icp,  # model tries to claim a good fit
        action=LeadAction.follow_up,
    )
    result = apply_icp_fit(classification)
    assert result.icp_verdict == ICPVerdict.out_of_icp
    assert result.action == LeadAction.ignore
