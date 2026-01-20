from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .models import EnrichedLeadClassification, HubSpotLead, LeadClassification
from .prompts import get_prompt_manager


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
    instructions: str | None = None,
    extra_tools: tuple[Callable, ...] | None = None,
) -> Agent[None, LeadClassification]:
    """
    Create a classification agent.

    Args:
        llm_base_url: Base URL for the LLM API
        llm_model_name: Model name to use
        llm_api_key: API key for the LLM
        instructions: System prompt (uses prompt manager if not provided)
        extra_tools: Additional tools for the agent (tuple for hashability)
    """
    provider = OpenAIProvider(base_url=llm_base_url, api_key=llm_api_key)
    model = OpenAIChatModel(model_name=llm_model_name, provider=provider)

    # Use provided instructions or get from prompt manager
    if instructions is None:
        instructions = get_prompt_manager().build_classification_prompt()

    tools = list(extra_tools) if extra_tools else []

    return Agent(
        model=model,
        output_type=LeadClassification,
        instructions=instructions,
        retries=2,
        end_strategy="early",
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=5000),
        tools=tools,
    )


def _create_research_agent(
    llm_base_url: str,
    llm_model_name: str,
    llm_api_key: str = "ollama",
    instructions: str | None = None,
) -> Agent[None, EnrichedLeadClassification]:
    """
    Create a research agent with web search capability.

    Args:
        llm_base_url: Base URL for the LLM API
        llm_model_name: Model name to use
        llm_api_key: API key for the LLM
        instructions: System prompt (uses prompt manager if not provided)
    """
    provider = OpenAIProvider(base_url=llm_base_url, api_key=llm_api_key)
    model = OpenAIChatModel(model_name=llm_model_name, provider=provider)

    # Use provided instructions or get from prompt manager
    if instructions is None:
        instructions = get_prompt_manager().build_research_prompt()

    return Agent(
        model=model,
        output_type=EnrichedLeadClassification,
        instructions=instructions,
        retries=2,
        end_strategy="early",
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=8000),
        tools=[duckduckgo_search_tool()],
    )


def classify_lead(
    settings: Settings,
    lead: HubSpotLead,
    *,
    debug: bool = False,
    enrich: bool = False,
    max_searches: int = 4,
) -> LeadClassification | EnrichedLeadClassification | ClassificationResult:
    """
    Classify a HubSpot lead using the LLM agent.

    Args:
        settings: Application settings with LLM config
        lead: Parsed HubSpot lead data
        debug: If True, return ClassificationResult with full message history
        enrich: If True, research promising leads with web search
        max_searches: Maximum number of web searches for enrichment (default 4)

    Returns:
        LeadClassification (or EnrichedLeadClassification if enrich=True),
        or ClassificationResult if debug=True
    """
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "ollama"

    # Get the current prompt from prompt manager (allows runtime updates)
    prompt_manager = get_prompt_manager()
    classification_prompt = prompt_manager.build_classification_prompt()

    agent = agent_factory(
        settings.llm_base_url,
        settings.llm_model_name,
        api_key,
        instructions=classification_prompt,
    )

    # Build prompt from lead data
    prompt = lead.to_prompt_text()
    result = agent.run_sync(prompt)
    classification = result.output

    # If promising and enrichment requested, do research
    if enrich and classification.label.value == "promising":
        enriched = _research_lead(settings, lead, classification, max_searches=max_searches)
        if debug:
            return ClassificationResult(
                classification=enriched,
                message_history=result.all_messages(),
                usage={
                    "request_tokens": getattr(result.usage(), "request_tokens", None),
                    "response_tokens": getattr(result.usage(), "response_tokens", None),
                    "total_tokens": getattr(result.usage(), "total_tokens", None),
                },
            )
        return enriched

    if debug:
        return ClassificationResult(
            classification=classification,
            message_history=result.all_messages(),
            usage={
                "request_tokens": getattr(result.usage(), "request_tokens", None),
                "response_tokens": getattr(result.usage(), "response_tokens", None),
                "total_tokens": getattr(result.usage(), "total_tokens", None),
            },
        )

    return classification


def _research_lead(
    settings: Settings,
    lead: HubSpotLead,
    classification: LeadClassification,
    max_searches: int = 4,
) -> EnrichedLeadClassification:
    """
    Research a promising lead using web search.

    Args:
        settings: Application settings
        lead: Original lead data
        classification: Initial classification result
        max_searches: Maximum number of searches to perform

    Returns:
        EnrichedLeadClassification with research findings
    """
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "ollama"

    # Get the current research prompt from prompt manager
    prompt_manager = get_prompt_manager()
    research_prompt = prompt_manager.build_research_prompt()

    # Create research agent
    research_agent = _create_research_agent(
        settings.llm_base_url,
        settings.llm_model_name,
        api_key,
        instructions=research_prompt,
    )

    # Build research prompt with lead context
    email_domain = ""
    if lead.email and "@" in lead.email:
        email_domain = lead.email.split("@")[1]

    company = classification.company or lead.company or email_domain
    contact_name = f"{lead.first_name or ''} {lead.last_name or ''}".strip()

    research_prompt = f"""
Research this promising lead:

Contact: {contact_name}
Email: {lead.email or "Unknown"}
Company: {company}
Email Domain: {email_domain}

Their message: {lead.message or lead.raw_text}

Initial classification:
- Label: {classification.label.value}
- Confidence: {classification.confidence:.0%}
- Reason: {classification.reason}

Please research:
1. First search the email domain ({email_domain}) to find the company website
2. Then search more broadly for "{company}" to understand what they do
3. Search for "{contact_name} {company}" to find their role/title

Limit yourself to {max_searches} total searches.
Return the enriched classification with your research findings.
"""

    try:
        result = research_agent.run_sync(research_prompt)
        return result.output
    except Exception as e:
        print(f"[RESEARCH] Error during research: {e}")
        # Return classification as enriched without research on error
        return EnrichedLeadClassification(
            first_name=classification.first_name,
            last_name=classification.last_name,
            email=classification.email,
            company=classification.company,
            label=classification.label,
            confidence=classification.confidence,
            reason=classification.reason,
            research_summary=f"Research failed: {e}",
        )


def classify_message(
    settings: Settings,
    text: str,
    *,
    debug: bool = False,
    enrich: bool = False,
    max_searches: int = 4,
) -> LeadClassification | EnrichedLeadClassification | ClassificationResult:
    """
    Classify a raw message text using the LLM agent.

    This is a convenience wrapper that creates a HubSpotLead from raw text.

    Args:
        settings: Application settings with LLM config
        text: Raw message text to classify
        debug: If True, return ClassificationResult with full message history
        enrich: If True, research promising leads with web search
        max_searches: Maximum number of web searches for enrichment

    Returns:
        LeadClassification (or EnrichedLeadClassification if enrich=True),
        or ClassificationResult if debug=True
    """
    # Create a simple lead with just the raw text
    lead = HubSpotLead(raw_text=text, message=text)
    return classify_lead(settings, lead, debug=debug, enrich=enrich, max_searches=max_searches)
