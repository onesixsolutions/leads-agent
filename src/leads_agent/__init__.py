"""leads-agent: AI-powered Slack lead classifier."""

__version__ = "0.1.0"

from leads_agent.models import LeadClassification, LeadLabel
from leads_agent.prompts import ICPConfig, PromptConfig, PromptManager

__all__ = [
    "LeadClassification",
    "LeadLabel",
    "PromptConfig",
    "ICPConfig",
    "PromptManager",
    "__version__",
]
