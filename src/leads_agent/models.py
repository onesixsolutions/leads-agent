from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LeadLabel(str, Enum):
    ignore = "ignore"
    promising = "promising"


class LeadAction(str, Enum):
    """Suggested action to take on this lead."""

    ignore = "ignore"
    follow_up = "follow_up"


class CriterionStatus(str, Enum):
    """
    Objective status of a single ICP criterion.

    `unknown` is a first-class answer and must be used whenever the evidence
    does not support a determination — it is never a synonym for `not_met`.
    """

    met = "met"
    not_met = "not_met"
    partial = "partial"
    unknown = "unknown"


class ICPVerdict(str, Enum):
    """
    Overall ICP determination. Derived deterministically from the criteria in
    `leads_agent.icp_fit` — never chosen directly by the model.
    """

    in_icp = "in_icp"
    partial_fit = "partial_fit"
    needs_verification = "needs_verification"
    out_of_icp = "out_of_icp"
    not_evaluated = "not_evaluated"


class HubSpotLead(BaseModel):
    """Parsed lead data from HubSpot Slack message."""

    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    company: str | None = None
    message: str | None = None
    raw_text: str = ""

    @classmethod
    def from_slack_event(cls, event: dict[str, Any]) -> HubSpotLead | None:
        """
        Parse a HubSpot bot message from Slack event.

        Returns None if this isn't a HubSpot message.
        """
        # Must be a bot_message from HubSpot
        if event.get("subtype") != "bot_message":
            return None
        if event.get("username", "").lower() != "hubspot":
            return None

        # Get text from attachments (HubSpot puts lead data there)
        attachments = event.get("attachments", [])
        if not attachments:
            return None

        # Use fallback or text from first attachment
        attachment = attachments[0]
        raw_text = attachment.get("fallback") or attachment.get("text") or ""

        if not raw_text:
            return None

        return cls._parse_hubspot_text(raw_text)

    @classmethod
    def _parse_hubspot_text(cls, text: str) -> HubSpotLead:
        """Parse HubSpot formatted text to extract lead fields."""
        lead = cls(raw_text=text)

        # Pattern: *Field Name*: Value
        # Handle both plain text and Slack markdown links like <mailto:email|email>
        #
        # Single-line fields must not cross a newline: with re.DOTALL an empty
        # field (e.g. "*Company*:" with nothing after it) would otherwise
        # swallow the following line, silently loading the message text into
        # `company` and sending the research stage after a bogus entity.
        single_line_patterns = {
            "first_name": r"\*First Name\*:[ \t]*([^\n]*)",
            "last_name": r"\*Last Name\*:[ \t]*([^\n]*)",
            "email": r"\*Email\*:[ \t]*(?:<mailto:[^|]+\|)?([^\s>]*)",
            "company": r"\*Company\*:[ \t]*([^\n]*)",
        }
        # The message is the one genuinely multi-line field.
        multi_line_patterns = {"message": r"\*Message\*:[ \t]*(.+)"}

        for field, pattern in single_line_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if not value:
                    continue
                value = re.sub(r"<mailto:[^|]+\|([^>]+)>", r"\1", value)
                value = re.sub(r"<[^|]+\|([^>]+)>", r"\1", value)
                setattr(lead, field, value)

        for field, pattern in multi_line_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if not value:
                    continue
                # Clean up the value
                value = re.sub(
                    r"<mailto:[^|]+\|([^>]+)>", r"\1", value
                )  # Clean email links
                value = re.sub(r"<[^|]+\|([^>]+)>", r"\1", value)  # Clean other links
                setattr(lead, field, value)

        return lead

    def to_prompt_text(self) -> str:
        """Format lead data for LLM prompt."""
        parts = []
        if self.first_name:
            parts.append(f"First Name: {self.first_name}")
        if self.last_name:
            parts.append(f"Last Name: {self.last_name}")
        if self.email:
            parts.append(f"Email: {self.email}")
        if self.company:
            parts.append(f"Company: {self.company}")
        if self.message:
            parts.append(f"Message: {self.message}")

        return "\n".join(parts) if parts else self.raw_text


class LeadClassification(BaseModel):
    """LLM output for lead classification with extracted contact info."""

    # Contact info (extracted/confirmed by LLM)
    first_name: str | None = Field(default=None, description="Contact's first name")
    last_name: str | None = Field(default=None, description="Contact's last name")
    email: str | None = Field(default=None, description="Contact's email address")
    company: str | None = Field(
        default=None, description="Contact's company name (if mentioned)"
    )

    # Classification
    label: LeadLabel = Field(description="Go/no-go decision: ignore or promising")
    reason: str = Field(description="Brief explanation for the classification")

    # Helpful synthesis to pass downstream (research/scoring)
    lead_summary: str | None = Field(
        default=None,
        description="1-2 sentence summary of the lead's intent and context (no fluff).",
    )
    key_signals: list[str] | None = Field(
        default=None,
        description="Short bullet-like signals (e.g., 'student project', 'budget mentioned', 'vendor pitch').",
    )


class CompanyResearch(BaseModel):
    """Research findings about a company."""

    company_name: str = Field(description="Official company name")
    company_description: str = Field(
        description="Brief description of what the company does"
    )
    industry: str | None = Field(default=None, description="Industry or sector")
    company_size: str | None = Field(
        default=None, description="Company size if found (revenue estimate and/or headcount)"
    )
    website: str | None = Field(default=None, description="Company website URL")
    relevance_notes: str | None = Field(
        default=None, description="Notes on why this lead might be relevant"
    )


class ContactResearch(BaseModel):
    """Research findings about a contact person."""

    full_name: str = Field(description="Contact's full name")
    title: str | None = Field(default=None, description="Job title or role")
    linkedin_summary: str | None = Field(
        default=None, description="Brief summary from LinkedIn or similar"
    )
    relevance_notes: str | None = Field(
        default=None, description="Notes on the contact's relevance"
    )


class ICPCriterion(BaseModel):
    """
    One objective ICP test.

    `finding` states what was determined; `evidence` states what it was
    determined *from*. A criterion with no supporting evidence must be
    `unknown`, not a guess.
    """

    status: CriterionStatus = Field(
        description="met / not_met / partial / unknown. Use unknown when evidence is absent."
    )
    finding: str = Field(
        description="One short, concrete sentence stating what was determined for this criterion."
    )
    evidence: str | None = Field(
        default=None,
        description="What the finding is based on (lead's own words, a research source, or an inference — say which). Null when status is unknown.",
    )


class ICPAssessment(BaseModel):
    """
    Fixed set of ICP criteria, evaluated for every non-spam lead.

    The set is fixed on purpose: the same criteria are reported for every lead
    so leads are comparable and nothing is quietly skipped.
    """

    company_footprint: ICPCriterion = Field(
        description=(
            "Could the COMPANY be identified and corroborated in search? met=found with "
            "corroborating detail. not_met=essentially no trace of the company exists — a "
            "red flag, because a $250M-$10B business has a web presence. unknown=research "
            "did not run. Judges the COMPANY only: a hard-to-find individual is normal and "
            "must never be scored here."
        )
    )
    revenue_band: ICPCriterion = Field(
        description="Is the company inside the $250M-$10B revenue band? met=in band, not_met=confidently below or above, unknown=no revenue evidence found."
    )
    executive_sponsor: ICPCriterion = Field(
        description="Is there a named executive sponsor with conviction and substantiated budget? not_met only when there is positive evidence of no sponsor/budget (e.g. a junior requester with no mandate)."
    )
    internal_team_depth: ICPCriterion = Field(
        description="Is the internal data/ML team thin relative to the pace the business demands? met=thin (good for us), not_met=large in-house build capability, unknown=no signal."
    )
    platform: ICPCriterion = Field(
        description="Platform trajectory. met=on Snowflake or migrating to it, partial=another modern cloud platform (Databricks/BigQuery/Fabric), not_met=legacy or on-prem with no modernization path, unknown=no signal."
    )
    entry_door: ICPCriterion = Field(
        description="Which of the six doors is this (Strategy & Governance, Modernization & Migration, Analytics & Activation, Agentic Automation, Predictive Automation, Decision Science)? not_met=no identifiable door, or enablement/workshop only with no build intent."
    )
    buyer_persona: ICPCriterion = Field(
        description="Which of the six personas is this contact, and does their title fit the company's revenue band? unknown=title not established."
    )
    trigger: ICPCriterion = Field(
        description="Is there a trigger creating urgency (AI mandate, platform migration, corporate event, recurring operational need)? unknown=none evident."
    )
    deal_shape: ICPCriterion = Field(
        description="Can this land at $150k+ as a new logo, or is it an existing-client expansion (floor-exempt)? not_met=explicitly below the floor as a new logo with no recurring path."
    )
    expansion_path: ICPCriterion = Field(
        description="Is there a credible path to a named phase two, retainer, Dedicated Data Team or operating partnership? unknown=not discussed."
    )
    focus_overlay: ICPCriterion = Field(
        description="Does the company sit in a focus overlay (Higher Education, Healthcare & Life Sciences, Manufacturing, Private Equity, Marketing function)? INFORMATIONAL ONLY - never gates the verdict, since the core ICP is horizontal."
    )


class OutreachBrief(BaseModel):
    """
    Short, outreach-ready brief.

    The criteria above are the objective layer. This is the judgement layer:
    the read a senior seller would give in a pipeline meeting, including the
    calls that the criteria cannot express on their own.
    """

    icp_statement: str = Field(
        description="The lead in one ICP-shaped sentence: revenue band, ownership/context, trigger, platform, door, likely entry shape. Say 'unknown' inline for anything not established rather than inventing it."
    )
    analyst_take: str = Field(
        description=(
            "2-4 sentences of plain-spoken judgement, the way an experienced seller would "
            "actually say it: what this looks like, what is off about it, what you would do. "
            "Inference is welcome here as long as you flag it as inference "
            "(e.g. 'reads like a seed-stage startup - gmail address, no company site, "
            "\"exploring options\" language')."
        )
    )
    opportunity: str | None = Field(
        default=None,
        description=(
            "Upside the immediate ask understates, when you see it — e.g. a small project "
            "that opens a much larger program, a beachhead into a PE portfolio, a "
            "referenceable proof point in a focus overlay. State the specific reason. "
            "Null when there is genuinely no upside beyond the ask."
        ),
    )
    risks: list[str] | None = Field(
        default=None,
        description="Concrete concerns: what could make this a bad engagement, or what looks off. Null if none.",
    )
    exception_case: str | None = Field(
        default=None,
        description=(
            "Only when the lead misses the ICP on the criteria but is still arguably worth "
            "pursuing as a NAMED exception — strategic logo, vertical proof point, PE portfolio "
            "wedge, or a credible path to a much larger program. State the named reason so a "
            "human can accept or reject it. Null when there is no such case: do not invent one "
            "to soften a rejection."
        ),
    )
    recommended_entry: str | None = Field(
        default=None,
        description="Recommended door plus entry shape (project archetype and rough size band).",
    )
    talking_points: list[str] | None = Field(
        default=None,
        description="2-4 concrete opening points grounded in the lead's own words or cited research. No invented facts.",
    )
    accelerators: list[str] | None = Field(
        default=None,
        description="Applicable OneSix accelerators for the identified door, if any.",
    )


class EnrichedLeadClassification(LeadClassification):
    """Lead classification enriched with web research and an ICP assessment."""

    # Research results (only populated for promising leads)
    company_research: CompanyResearch | None = Field(
        default=None, description="Research findings about the company"
    )
    contact_research: ContactResearch | None = Field(
        default=None, description="Research findings about the contact person"
    )
    research_summary: str | None = Field(
        default=None, description="Executive summary of research findings"
    )

    # ICP assessment — the model fills the criteria and the brief...
    icp_assessment: ICPAssessment | None = Field(
        default=None,
        description="Per-criterion ICP evaluation. Fill every criterion; use 'unknown' where evidence is absent.",
    )
    brief: OutreachBrief | None = Field(
        default=None,
        description="Outreach brief. Fill this in for any lead that is not spam.",
    )

    # ...and these are derived deterministically in icp_fit — the model
    # must not set them.
    icp_verdict: ICPVerdict | None = Field(
        default=None,
        description="DERIVED IN CODE — do not set. Overall ICP determination.",
    )
    reasons_in_icp: list[str] | None = Field(
        default=None,
        description="DERIVED IN CODE — do not set. Criteria that qualify this lead.",
    )
    reasons_out_of_icp: list[str] | None = Field(
        default=None,
        description="DERIVED IN CODE — do not set. Criteria that disqualify this lead.",
    )
    open_questions: list[str] | None = Field(
        default=None,
        description="DERIVED IN CODE — do not set. What must be verified before pursuing.",
    )
    action: LeadAction | None = Field(
        default=None,
        description="DERIVED IN CODE — do not set. Recommended action (ignore/follow_up).",
    )
