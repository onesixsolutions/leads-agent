"""
Tests for the paced DuckDuckGo wrapper.

DuckDuckGo throttles bursts, and an exception from the search tool used to
propagate out of the research agent and discard every finding for the lead.
These tests pin the three properties that prevent that: failures degrade to a
message, the call budget is hard, and calls are serialised.
"""

from __future__ import annotations

import asyncio

import pytest

from leads_agent import agent as agent_mod


class _FakeTool:
    """Stands in for pydantic-ai's DuckDuckGo tool."""

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(self, query: str):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append(query)
        try:
            await asyncio.sleep(0)
            return self._behaviour(len(self.calls))
        finally:
            self.concurrent -= 1


@pytest.fixture(autouse=True)
def _fast_pacing(monkeypatch):
    """Keep the sleeps out of the test suite."""
    monkeypatch.setattr(agent_mod, "_SEARCH_MIN_INTERVAL_S", 0.0)


def _install(monkeypatch, behaviour) -> _FakeTool:
    fake = _FakeTool(behaviour)
    monkeypatch.setattr(
        agent_mod, "duckduckgo_search_tool", lambda: type("T", (), {"function": fake})()
    )
    return fake


def test_successful_search_passes_results_through(monkeypatch):
    fake = _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("uline revenue"))
    assert result == [{"title": "hit"}]
    assert fake.calls == ["uline revenue"]


def test_raising_search_degrades_to_a_message_not_an_exception(monkeypatch):
    """The critical property: research must survive a throttled search."""
    def boom(n):
        raise RuntimeError("rate limited")

    _install(monkeypatch, boom)
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("anything"))
    assert isinstance(result, str)
    assert result.startswith("SEARCH_UNAVAILABLE")
    # And it must tell the model this is tooling, not evidence of absence.
    assert "TOOLING FAILURE" in result


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    fake = _install(monkeypatch, lambda n: [] if n < 3 else [{"title": "late hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("q"))
    assert result == [{"title": "late hit"}]
    assert len(fake.calls) == 3  # two empties, then a hit


def test_empty_results_exhaust_retries_and_report_unavailable(monkeypatch):
    fake = _install(monkeypatch, lambda n: [])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("q"))
    assert result.startswith("SEARCH_UNAVAILABLE")
    assert len(fake.calls) == agent_mod._SEARCH_MAX_ATTEMPTS


def test_search_budget_is_enforced(monkeypatch):
    """`max_searches` was previously only a suggestion in the prompt."""
    fake = _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=2)

    async def run_all():
        return [await tool.function(f"q{i}") for i in range(4)]

    results = asyncio.run(run_all())
    assert len(fake.calls) == 2, "budget exceeded"
    assert results[0] == [{"title": "hit"}]
    assert results[2].startswith("SEARCH_BUDGET_EXHAUSTED")
    assert results[3].startswith("SEARCH_BUDGET_EXHAUSTED")


def test_concurrent_calls_are_serialised(monkeypatch):
    """Parallel searches are what trip the rate limiter, so they must queue."""
    fake = _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=8)

    async def run_parallel():
        await asyncio.gather(*(tool.function(f"q{i}") for i in range(5)))

    asyncio.run(run_parallel())
    assert fake.max_concurrent == 1, f"searches ran {fake.max_concurrent}-way concurrent"
    assert len(fake.calls) == 5
