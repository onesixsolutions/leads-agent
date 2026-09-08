from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, overload

from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import Tool

from leads_agent.config import Settings
from leads_agent.icp_fit import apply_icp_fit
from leads_agent.models import EnrichedLeadClassification, HubSpotLead, LeadClassification
from leads_agent.prompts import get_prompt_manager

logger = logging.getLogger(__name__)

TOutput = TypeVar("TOutput")

# --- DuckDuckGo pacing -------------------------------------------------------
#
# DuckDuckGo rate-limits hard when an agent fires searches back to back, and a
# throttled call raises — which would otherwise propagate out of the research
# agent and discard every finding for that lead.
#
# The spacing state is deliberately MODULE-level, not per-tool: a fresh
# research agent (and therefore a fresh tool) is constructed for every lead, so
# per-instance state would reset between leads and the first search of lead N+1
# would fire immediately after the last search of lead N — exactly the burst
# that gets us throttled during a backtest.
_SEARCH_MIN_INTERVAL_S = 2.0        # steady-state spacing between any two searches
_SEARCH_MAX_ATTEMPTS = 4            # attempts per query when the tool errors
_SEARCH_BACKOFF_BASE_S = 2.0        # exponential: 2s, 4s, 8s ...
_SEARCH_BACKOFF_CAP_S = 30.0        # ceiling for any single backoff sleep
_SEARCH_RATELIMIT_FACTOR = 3.0      # a ratelimit needs longer than a timeout
_SEARCH_JITTER = 0.25               # +/- fraction, so parallel leads desynchronise

# Process-global clock for inter-search spacing. A bare float rather than an
# asyncio primitive on purpose: each Agent.run_sync() may run on its own event
# loop, and an asyncio.Lock shared across loops raises. Intra-run serialisation
# is handled by a per-instance lock (always one loop); this only enforces the
# minimum gap, where a rare lost update just means one slightly short sleep.
_last_search_at: float = 0.0

# Three distinct outcomes the model must be able to tell apart. Conflating the
# last two is what would break the company_footprint rule.
_SEARCH_BUDGET_SPENT = (
    "SEARCH_BUDGET_EXHAUSTED: no searches remain for this lead. Report what you "
    "have and mark anything still unestablished as unknown."
)
_SEARCH_NO_RESULTS = (
    "SEARCH_RETURNED_NO_RESULTS: the search ran successfully and genuinely "
    "returned zero results for this query. This IS evidence of absence for this "
    "query \u2014 if well-formed searches for the company name and domain all "
    "come back like this, that supports company_footprint = not_met."
)
_SEARCH_UNAVAILABLE = (
    "SEARCH_UNAVAILABLE: the search tool errored or was rate-limited and did not "
    "run. This is a TOOLING FAILURE and is NOT evidence about the company \u2014 "
    "never treat it as an absence of web presence. Mark the affected criteria "
    "unknown and say the research was unavailable."
)


@dataclass
class ClassificationResult:
    """Result of the triage/research/assessment pipeline with optional debug info."""

    classification: LeadClassification | EnrichedLeadClassification
    message_history: list[ModelMessage] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.classification.label.value

    @property
    def reason(self) -> str:
        return self.classification.reason

    @property
    def icp_verdict(self) -> str | None:
        verdict = getattr(self.classification, "icp_verdict", None)
        return verdict.value if verdict is not None else None

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
        print("LEAD PIPELINE DEBUG")
        print("=" * 60)
        print(f"Label: {self.label}")
        print(f"ICP verdict: {self.icp_verdict or 'n/a'}")
        print(f"Reason: {self.reason}")
        print(f"\nUsage: {self.usage}")
        print(f"\nMessage History ({len(self.message_history)} messages):")
        print(self.format_history(verbose=verbose))
        print("=" * 60 + "\n")


@overload
def agent_factory(
    *,
    llm_model_name: str,
    llm_api_key: str,
    instructions: str | None = None,
    output_type: type[TOutput],
    model_settings: AnthropicModelSettings,
    extra_tools: tuple[Callable, ...] | None = None,
    search_budget: int | None = None,
) -> Agent[None, TOutput]: ...


def agent_factory(
    *,
    llm_model_name: str,
    llm_api_key: str,
    instructions: str | None = None,
    output_type: type[TOutput],
    model_settings: AnthropicModelSettings,
    extra_tools: tuple[Callable, ...] | None = None,
    search_budget: int | None = None,
) -> Agent[None, TOutput]:
    """
    Create an agent in a consistent way across triage/research/assessment.
    """
    provider = AnthropicProvider(api_key=llm_api_key)
    model = AnthropicModel(model_name=llm_model_name, provider=provider)

    tools: list[Any] = list(extra_tools) if extra_tools else []
    if search_budget:
        tools.append(_paced_duckduckgo_tool(search_budget))

    return Agent(
        model=model,
        output_type=output_type,
        instructions=instructions or "",
        retries=2,
        end_strategy="early",
        model_settings=model_settings,
        tools=tools,
    )


def _backoff_delay(attempt: int, *, rate_limited: bool) -> float:
    """Exponential backoff with jitter, capped, longer when rate-limited."""
    delay = _SEARCH_BACKOFF_BASE_S * (2 ** (attempt - 1))
    if rate_limited:
        delay *= _SEARCH_RATELIMIT_FACTOR
    delay = min(delay, _SEARCH_BACKOFF_CAP_S)
    return delay * (1.0 + random.uniform(-_SEARCH_JITTER, _SEARCH_JITTER))


async def _await_search_slot() -> None:
    """Sleep until the process-global minimum gap since the last search."""
    gap = _SEARCH_MIN_INTERVAL_S - (time.monotonic() - _last_search_at)
    if gap > 0:
        await asyncio.sleep(gap)


def _paced_duckduckgo_tool(max_searches: int) -> Tool:
    """
    DuckDuckGo search with global pacing, exponential backoff and a hard budget.

    Guarantees for the caller:
    - calls are serialised within a run and spaced across the whole process,
    - transient errors are retried with exponential backoff (longer for a
      rate-limit) and never escape as exceptions,
    - a genuine empty result is reported differently from a tool failure, so
      the company_footprint rule keeps working.
    """
    inner = duckduckgo_search_tool()
    search = inner.function
    lock = asyncio.Lock()
    state = {"calls": 0}

    async def duckduckgo_search(query: str) -> Any:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        global _last_search_at

        # Serialising matters: concurrent calls are what trip the limiter.
        async with lock:
            if state["calls"] >= max_searches:
                return _SEARCH_BUDGET_SPENT
            state["calls"] += 1

            saw_empty = False
            for attempt in range(1, _SEARCH_MAX_ATTEMPTS + 1):
                await _await_search_slot()

                rate_limited = False
                try:
                    results = await search(query)
                except RatelimitException as exc:
                    rate_limited = True
                    results = None
                    logger.warning(
                        "duckduckgo rate-limited (attempt %d/%d) for %r: %s",
                        attempt, _SEARCH_MAX_ATTEMPTS, query, exc,
                    )
                except (TimeoutException, DDGSException) as exc:
                    results = None
                    logger.warning(
                        "duckduckgo error (attempt %d/%d) for %r: %s",
                        attempt, _SEARCH_MAX_ATTEMPTS, query, exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Anything else (network, parsing, validation) is retryable
                    # too; it must never escape and kill the research stage.
                    results = None
                    logger.warning(
                        "duckduckgo unexpected failure (attempt %d/%d) for %r: %s",
                        attempt, _SEARCH_MAX_ATTEMPTS, query, exc,
                    )
                finally:
                    _last_search_at = time.monotonic()

                if results:
                    return results

                if results is not None:
                    # Ran fine but returned nothing. A soft throttle can look
                    # like this, so retry once; a second empty is taken as a
                    # real absence rather than being misreported as a failure.
                    if saw_empty:
                        logger.info("duckduckgo returned no results for %r", query)
                        return _SEARCH_NO_RESULTS
                    saw_empty = True

                if attempt < _SEARCH_MAX_ATTEMPTS:
                    await asyncio.sleep(_backoff_delay(attempt, rate_limited=rate_limited))

            if saw_empty:
                return _SEARCH_NO_RESULTS
            logger.warning("duckduckgo exhausted %d attempts for %r", _SEARCH_MAX_ATTEMPTS, query)
            return _SEARCH_UNAVAILABLE

    return Tool(duckduckgo_search, name="duckduckgo_search", takes_ctx=False)


# Effort per stage. Triage is a cheap spam filter; the ICP assessment is where
# the judgement calls happen and is worth the extra thinking.
_STAGE_EFFORT: dict[str, str] = {
    "triage": "low",
    "research": "high",
    "assessment": "xhigh",
}


def _model_settings(settings: Settings, stage: str) -> AnthropicModelSettings:
    """
    Build per-stage model settings.

    Note: `temperature`, `top_p`, `top_k` and thinking `budget_tokens` are all
    rejected (HTTP 400) by the Opus 5 / Sonnet 5 generation. Depth is controlled
    with adaptive thinking plus `anthropic_effort` instead.
    """
    return AnthropicModelSettings(
        max_tokens=settings.llm_max_tokens,
        anthropic_thinking={"type": "adaptive"},
        anthropic_effort=_STAGE_EFFORT.get(stage, "high"),
    )


def _usage_snapshot(result: Any) -> dict[str, Any]:
    """Best-effort extraction of token usage from pydantic-ai result."""
    # `usage` is a property in pydantic-ai >=1.x; tolerate the older callable.
    try:
        usage = result.usage
        if callable(usage):
            usage = usage()
    except Exception:
        usage = None
    return {
        "request_tokens": getattr(usage, "request_tokens", None) if usage is not None else None,
        "response_tokens": getattr(usage, "response_tokens", None) if usage is not None else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
    }


def _create_triage_agent(settings: Settings, api_key: str) -> Agent[None, LeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_triage_prompt(),
        output_type=LeadClassification,
        model_settings=_model_settings(settings, "triage"),
    )


def _create_research_agent(
    settings: Settings, api_key: str, max_searches: int = 4
) -> Agent[None, EnrichedLeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_research_prompt(),
        output_type=EnrichedLeadClassification,
        model_settings=_model_settings(settings, "research"),
        search_budget=max_searches,
    )


def _create_assessment_agent(settings: Settings, api_key: str) -> Agent[None, EnrichedLeadClassification]:
    pm = get_prompt_manager()
    return agent_factory(
        llm_model_name=settings.llm_model_name,
        llm_api_key=api_key,
        instructions=pm.build_icp_assessment_prompt(),
        output_type=EnrichedLeadClassification,
        model_settings=_model_settings(settings, "assessment"),
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
    triage → (if promising) web research → (if promising) ICP assessment.

    The ICP verdict is derived from the assessment's criteria in
    `icp_fit`, not chosen by the model.
    """
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""

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

        scored, assessment_msgs, assessment_usage = _assess_lead(
            settings,
            lead,
            triage=triage,
            enriched=enriched,
            return_debug=True,
        )
        final = scored
        if assessment_msgs:
            message_history.extend(assessment_msgs)
        if assessment_usage:
            usage["assessment"] = assessment_usage

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
) -> EnrichedLeadClassification | tuple[EnrichedLeadClassification, list[ModelMessage], dict[str, Any]]:
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    research_agent = _create_research_agent(settings, api_key, max_searches=max_searches)

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
            reason=classification.reason,
            lead_summary=classification.lead_summary,
            key_signals=classification.key_signals,
            research_summary=f"Research failed: {e}",
        )
        if return_debug:
            return apply_icp_fit(fallback), [], {"error": str(e)}
        return apply_icp_fit(fallback)


def _assess_lead(
    settings: Settings,
    lead: HubSpotLead,
    *,
    triage: LeadClassification,
    enriched: EnrichedLeadClassification | None,
    return_debug: bool = False,
) -> EnrichedLeadClassification | tuple[EnrichedLeadClassification, list[ModelMessage], dict[str, Any]]:
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    assessment_agent = _create_assessment_agent(settings, api_key)

    name = f"{lead.first_name or ''} {lead.last_name or ''}".strip()
    email_domain = ""
    if lead.email and "@" in lead.email:
        email_domain = lead.email.split("@")[1]

    assessment_input = f"""
Lead:
- Name: {name or "Unknown"}
- Email: {lead.email or "Unknown"} (domain: {email_domain or "Unknown"})
- Company (parsed): {lead.company or "Unknown"}
- Message: {lead.message or lead.raw_text}

Triage output:
- label: {triage.label.value}
- reason: {triage.reason}
- lead_summary: {triage.lead_summary or "N/A"}
- key_signals: {", ".join(triage.key_signals or []) or "N/A"}
- extracted_company: {triage.company or "N/A"}

Research output (if any):
{enriched.model_dump_json(indent=2, exclude_none=True) if enriched is not None else "None"}
"""

    run = assessment_agent.run_sync(assessment_input)
    # The model supplies criteria + brief; the verdict/action are derived here so
    # identical evidence always yields an identical decision.
    output = apply_icp_fit(run.output)
    if return_debug:
        return output, run.all_messages(), _usage_snapshot(run)
    return output


def triage_lead(settings: Settings, lead: HubSpotLead) -> LeadClassification:
    """Run only the triage stage — no research or scoring."""
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    agent = _create_triage_agent(settings, api_key)
    run = agent.run_sync(lead.to_prompt_text())
    return run.output


def enrich_lead(
    settings: Settings,
    lead: HubSpotLead,
    triage: LeadClassification,
    *,
    max_searches: int = 4,
) -> EnrichedLeadClassification:
    """Run research + ICP assessment on a promising lead (call after triage_lead)."""
    enriched, _, _ = _research_lead(settings, lead, triage, max_searches=max_searches, return_debug=True)
    scored, _, _ = _assess_lead(settings, lead, triage=triage, enriched=enriched, return_debug=True)
    return scored


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
