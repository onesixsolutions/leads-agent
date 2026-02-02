from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, overload

import logfire
from opentelemetry import trace
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from leads_agent.config import Settings
from leads_agent.models import (
    EnrichedLeadClassification,
    HubSpotLead,
    LeadClassification,
)
from leads_agent.prompts import get_prompt_manager

# Configure logfire only if token is available
_logfire_enabled = bool(os.environ.get("LOGFIRE_TOKEN"))
if _logfire_enabled:
    try:
        logfire.configure()
        logfire.instrument_pydantic_ai()
    except Exception:
        # If configuration fails, disable logfire
        _logfire_enabled = False


@contextmanager
def _logfire_span(name: str, **kwargs):
    """Context manager for logfire spans that works even when logfire is disabled."""
    if _logfire_enabled:
        with logfire.span(name, **kwargs):
            yield
    else:
        yield


TOutput = TypeVar("TOutput")


@dataclass
class ClassificationResult:
    """Result of classification with optional debug info."""

    classification: LeadClassification | EnrichedLeadClassification
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
                        lines.append(
                            f"  └─ {part_type}: {part.tool_name}({getattr(part, 'args', {})})"
                        )
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


@overload
def agent_factory(
    *,
    llm_base_url: str,
    llm_model_name: str,
    llm_api_key: str = "ollama",
    instructions: str | None = None,
    output_type: type[TOutput],
    model_settings: OpenAIChatModelSettings,
    extra_tools: tuple[Callable, ...] | None = None,
    use_duckduckgo_search: bool = False,
) -> Agent[None, TOutput]: ...


def agent_factory(
    *,
    llm_base_url: str,
    llm_model_name: str,
    llm_api_key: str = "ollama",
    instructions: str | None = None,
    output_type: type[TOutput],
    model_settings: OpenAIChatModelSettings,
    extra_tools: tuple[Callable, ...] | None = None,
    use_duckduckgo_search: bool = False,
) -> Agent[None, TOutput]:
    """
    Create an agent in a consistent way across triage/research/scoring.
    """
    provider = OpenAIProvider(base_url=llm_base_url, api_key=llm_api_key)
    model = OpenAIChatModel(model_name=llm_model_name, provider=provider)

    tools: list[Any] = list(extra_tools) if extra_tools else []
    if use_duckduckgo_search:
        tools.append(duckduckgo_search_tool())

    return Agent(
        model=model,
        output_type=output_type,
        instructions=instructions or "",
        retries=2,
        end_strategy="early",
        model_settings=model_settings,
        tools=tools,
    )


def _usage_snapshot(result: Any) -> dict[str, Any]:
    """Best-effort extraction of token usage from pydantic-ai result."""
    try:
        usage = result.usage()
    except Exception:
        usage = None
    return {
        "request_tokens": getattr(usage, "request_tokens", None)
        if usage is not None
        else None,
        "response_tokens": getattr(usage, "response_tokens", None)
        if usage is not None
        else None,
        "total_tokens": getattr(usage, "total_tokens", None)
        if usage is not None
        else None,
    }


def _create_triage_agent(
    settings: Settings, api_key: str
) -> Agent[None, LeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_base_url=settings.llm_base_url,
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_triage_prompt(),
        output_type=LeadClassification,
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=900),
    )


def _create_research_agent(
    settings: Settings, api_key: str
) -> Agent[None, EnrichedLeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_base_url=settings.llm_base_url,
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_research_prompt(),
        output_type=EnrichedLeadClassification,
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=8000),
        use_duckduckgo_search=True,
    )


def _create_scoring_agent(
    settings: Settings, api_key: str
) -> Agent[None, EnrichedLeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_base_url=settings.llm_base_url,
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_scoring_prompt(),
        output_type=EnrichedLeadClassification,
        model_settings=OpenAIChatModelSettings(temperature=0.0, max_tokens=2500),
    )


def classify_lead(
    settings: Settings,
    lead: HubSpotLead,
    *,
    debug: bool = False,
    max_searches: int = 4,
) -> LeadClassification | EnrichedLeadClassification | ClassificationResult:
    """
    Classify a HubSpot lead using a multi-stage pipeline:
    triage → (if promising) web research → (if promising) final 1–5 scoring.
    """
    # Ensure there is a stable parent span even when classify_lead is called directly
    # (e.g., CLI/backtest). When invoked under an existing span (e.g., Slack processing),
    # we create a child span instead.
    current = trace.get_current_span()
    has_parent = current.get_span_context().is_valid

    lead_id = ""
    if lead.email:
        lead_id = lead.email.lower()
    if not lead_id:
        base = "|".join(
            [
                lead.company or "",
                lead.first_name or "",
                lead.last_name or "",
                (lead.message or lead.raw_text or "")[:500],
            ]
        )
        lead_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

    span_name = "lead.classify" if has_parent else "lead.process"
    with _logfire_span(
        span_name,
        lead_id=lead_id,
        email=lead.email,
        company=lead.company,
        max_searches=max_searches,
    ):
        api_key = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else "ollama"
        )

        triage_agent = _create_triage_agent(settings, api_key)
        prompt = lead.to_prompt_text()
        triage_run = triage_agent.run_sync(prompt)
        triage = triage_run.output

        final: LeadClassification | EnrichedLeadClassification = triage
        message_history: list[ModelMessage] = []
        usage: dict[str, Any] = {"triage": _usage_snapshot(triage_run)}
        try:
            message_history.extend(triage_run.all_messages())
        except Exception:
            pass

        if triage.label.value == "promising":
            enriched, research_msgs, research_usage = _research_lead(
                settings, lead, triage, max_searches=max_searches, return_debug=True
            )
            if research_msgs:
                message_history.extend(research_msgs)
            if research_usage:
                usage["research"] = research_usage

            scored, scoring_msgs, scoring_usage = _score_lead(
                settings,
                lead,
                triage=triage,
                enriched=enriched,
                return_debug=True,
            )
            final = scored
            if scoring_msgs:
                message_history.extend(scoring_msgs)
            if scoring_usage:
                usage["scoring"] = scoring_usage

        if debug:
            return ClassificationResult(
                classification=final,
                message_history=message_history,
                usage=usage,
            )
        return final


def _research_lead(
    settings: Settings,
    lead: HubSpotLead,
    classification: LeadClassification,
    max_searches: int = 4,
    return_debug: bool = False,
) -> (
    EnrichedLeadClassification
    | tuple[EnrichedLeadClassification, list[ModelMessage], dict[str, Any]]
):
    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key
        else "ollama"
    )
    research_agent = _create_research_agent(settings, api_key)

    email_domain = ""
    if lead.email and "@" in lead.email:
        email_domain = lead.email.split("@")[1]

    company = classification.company or lead.company or email_domain
    contact_name = f"{lead.first_name or ''} {lead.last_name or ''}".strip()

    research_prompt = f"""
Research this promising lead:

Contact: {contact_name or "Unknown"}
Email: {lead.email or "Unknown"}
Company (best guess): {company or "Unknown"}
Email Domain: {email_domain or "Unknown"}

Lead summary (triage): {classification.lead_summary or "N/A"}
Key signals (triage): {", ".join(classification.key_signals or []) or "N/A"}

Original message:
{lead.message or lead.raw_text}

Triage classification:
- Label: {classification.label.value}
- Confidence: {classification.confidence:.0%}
- Reason: {classification.reason}

Research plan:
1) If an email domain is present ({email_domain or "N/A"}), search it to identify the official website and company name.
2) Search the company name to understand what they do (quick description, industry, size if available).
3) Search "{contact_name} {company}" to find role/title (if name/company are available).

Query quality requirements:
- Use DuckDuckGo operators where helpful (quotes, site:, exclusions like -jobs -careers, and small OR groups).
- Use the "Query Operator Clause Pack" provided in your system prompt to add ICP/focus-area qualifiers.
- Before each tool call, draft 2–3 candidate queries, then pick the best one.

Limit yourself to {max_searches} total searches.
Return an enriched classification with your research findings.
"""

    try:
        run = research_agent.run_sync(research_prompt)
        output = run.output
        if return_debug:
            return output, run.all_messages(), _usage_snapshot(run)
        return output
    except Exception as e:
        fallback = EnrichedLeadClassification(
            first_name=classification.first_name,
            last_name=classification.last_name,
            email=classification.email,
            company=classification.company,
            label=classification.label,
            confidence=classification.confidence,
            reason=classification.reason,
            lead_summary=classification.lead_summary,
            key_signals=classification.key_signals,
            research_summary=f"Research failed: {e}",
        )
        if return_debug:
            return fallback, [], {"error": str(e)}
        return fallback


def _score_lead(
    settings: Settings,
    lead: HubSpotLead,
    *,
    triage: LeadClassification,
    enriched: EnrichedLeadClassification | None,
    return_debug: bool = False,
) -> (
    EnrichedLeadClassification
    | tuple[EnrichedLeadClassification, list[ModelMessage], dict[str, Any]]
):
    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key
        else "ollama"
    )
    scoring_agent = _create_scoring_agent(settings, api_key)

    name = f"{lead.first_name or ''} {lead.last_name or ''}".strip()
    email_domain = ""
    if lead.email and "@" in lead.email:
        email_domain = lead.email.split("@")[1]

    scoring_input = f"""
Lead:
- Name: {name or "Unknown"}
- Email: {lead.email or "Unknown"} (domain: {email_domain or "Unknown"})
- Company (parsed): {lead.company or "Unknown"}
- Message: {lead.message or lead.raw_text}

Triage output:
- label: {triage.label.value}
- confidence: {triage.confidence:.0%}
- reason: {triage.reason}
- lead_summary: {triage.lead_summary or "N/A"}
- key_signals: {", ".join(triage.key_signals or []) or "N/A"}
- extracted_company: {triage.company or "N/A"}

Research output (if any):
{enriched.model_dump_json(indent=2, exclude_none=True) if enriched is not None else "None"}
"""

    run = scoring_agent.run_sync(scoring_input)
    output = run.output
    if return_debug:
        return output, run.all_messages(), _usage_snapshot(run)
    return output


def classify_message(
    settings: Settings,
    text: str,
    *,
    debug: bool = False,
    max_searches: int = 4,
) -> LeadClassification | EnrichedLeadClassification | ClassificationResult:
    """Classify a raw message text using the same pipeline as classify_lead()."""
    lead = HubSpotLead(raw_text=text, message=text)
    return classify_lead(settings, lead, debug=debug, max_searches=max_searches)
