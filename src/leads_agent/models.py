from enum import Enum

from pydantic import BaseModel, Field


class LeadLabel(str, Enum):
    spam = "spam"
    solicitation = "solicitation"
    promising = "promising"


class LeadClassification(BaseModel):
    label: LeadLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

