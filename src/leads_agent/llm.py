from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .models import HubSpotLead, LeadClassification

SYSTEM_PROMPT = """\
You classify inbound leads from a consulting company contact form.

You will receive lead information including name, email, and their message.
Extract and return the contact details along with your classification.

Classification labels:
- spam: irrelevant, automated, SEO/link-building, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnership offers
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative — if unclear, choose spam
- Extract the company name from the message or email domain if not provided
- Provide a brief reason for your classification
"""


@dataclass
class ClassificationResult:
    """Result of classification with optional debug info."""

    classification: LeadClassification
    message_history: list[ModelMessage] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.classification.label.value

    @property
    def confidence(self) -> float:
        return self.classification.confidence

    @property
    def reason(self) -> str:
        return self.classification.reason

    def format_history(self, verbose: bool = False) -> str:
        """Format message history for debugging output."""
        lines = []
        for i, msg in enumerate(self.message_history):
            msg_type = type(msg).__name__
            lines.append(f"\n[{i}] {msg_type}")

            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = type(part).__name__
                    if hasattr(part, "content"):
                        content = part.content
                        if not verbose and len(str(content)) > 200:
                            content = str(content)[:200] + "..."
                        lines.append(f"  └─ {part_type}: {content}")
                    elif hasattr(part, "tool_name"):
                        lines.append(f"  └─ {part_type}: {part.tool_name}({getattr(part, 'args', {})})")
                    else:
                        lines.append(f"  └─ {part_type}: {part}")
            else:
                lines.append(f"  └─ {msg}")

        return "\n".join(lines)

    def print_debug(self, verbose: bool = False) -> None:
        """Print debug information to console."""
        print("\n" + "=" * 60)
        print("CLASSIFICATION DEBUG")
        print("=" * 60)
        print(f"Label: {self.label}")
        print(f"Confidence: {self.confidence:.2%}")
        print(f"Reason: {self.reason}")
        print(f"\nUsage: {self.usage}")
        print(f"\nMessage History ({len(self.message_history)} messages):")
        print(self.format_history(verbose=verbose))
        print("=" * 60 + "\n")


@lru_cache(maxsize=8)
def agent_factory(
    llm_base_url: str,
    llm_model_name: str,
    llm_api_key: str = "ollama",
    instructions: str = SYSTEM_PROMPT,
    extra_tools: list[Callable] | None = None,
) -> Agent[None, LeadClassification]:
    provider = OpenAIProvider(base_url=llm_base_url, api_key=llm_api_key)
    model = OpenAIChatModel(model_name=llm_model_name, provider=provider)

    tools = [] + (extra_tools or [])

    return Agent(
        model=model,
        output_type=LeadClassification,
        instructions=instructions,
        retries=2,
        end_strategy="early",
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=5000),
        tools=tools,
    )


def classify_lead(
    settings: Settings, lead: HubSpotLead, *, debug: bool = False
) -> LeadClassification | ClassificationResult:
    """
    Classify a HubSpot lead using the LLM agent.

    Args:
        settings: Application settings with LLM config
        lead: Parsed HubSpot lead data
        debug: If True, return ClassificationResult with full message history

    Returns:
        LeadClassification if debug=False, ClassificationResult if debug=True
    """
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "ollama"
    agent = agent_factory(
        settings.llm_base_url,
        settings.llm_model_name,
        api_key,
    )

    # Build prompt from lead data
    prompt = lead.to_prompt_text()
    result = agent.run_sync(prompt)

    if debug:
        return ClassificationResult(
            classification=result.output,
            message_history=result.all_messages(),
            usage={
                "request_tokens": getattr(result.usage(), "request_tokens", None),
                "response_tokens": getattr(result.usage(), "response_tokens", None),
                "total_tokens": getattr(result.usage(), "total_tokens", None),
            },
        )

    return result.output


def classify_message(
    settings: Settings, text: str, *, debug: bool = False
) -> LeadClassification | ClassificationResult:
    """
    Classify a raw message text using the LLM agent.

    This is a convenience wrapper that creates a HubSpotLead from raw text.

    Args:
        settings: Application settings with LLM config
        text: Raw message text to classify
        debug: If True, return ClassificationResult with full message history

    Returns:
        LeadClassification if debug=False, ClassificationResult if debug=True
    """
    # Create a simple lead with just the raw text
    lead = HubSpotLead(raw_text=text, message=text)
    return classify_lead(settings, lead, debug=debug)
