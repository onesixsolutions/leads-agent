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
    prioritize = "prioritize"


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
        patterns = {
            "first_name": r"\*First Name\*:\s*(.+?)(?=\n\*|\n*$)",
            "last_name": r"\*Last Name\*:\s*(.+?)(?=\n\*|\n*$)",
            "email": r"\*Email\*:\s*(?:<mailto:[^|]+\|)?([^\s>]+)",
            "company": r"\*Company\*:\s*(.+?)(?=\n\*|\n*$)",
            "message": r"\*Message\*:\s*(.+)",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
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
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
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
        default=None, description="Company size if found (startup, SMB, enterprise)"
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


class EnrichedLeadClassification(LeadClassification):
    """Lead classification enriched with web research for promising leads."""

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

    # Scoring results (only populated when we do scoring; typically for promising leads)
    score: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Lead score 1-5 where 1=not worth pursuing and 5=high-intent + strong ICP fit.",
    )
    action: LeadAction | None = Field(
        default=None,
        description="Recommended action (ignore/follow_up/prioritize).",
    )
    score_reason: str | None = Field(
        default=None,
        description="Brief explanation for the score/action decision.",
    )
