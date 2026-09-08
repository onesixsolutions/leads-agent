"""
Tests for the paced DuckDuckGo wrapper.

Fully offline: the search function is faked, and sleeps are captured rather
than performed. Each test pins one property that a real rate-limit would
otherwise break.
"""

from __future__ import annotations

import asyncio

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from leads_agent import agent as agent_mod


class _FakeSearch:
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


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Capture every sleep instead of actually waiting."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float):
        recorded.append(seconds)

    monkeypatch.setattr(agent_mod.asyncio, "sleep", fake_sleep)
    return recorded


@pytest.fixture(autouse=True)
def _reset_global_clock(monkeypatch):
    monkeypatch.setattr(agent_mod, "_last_search_at", 0.0)


def _install(monkeypatch, behaviour) -> _FakeSearch:
    fake = _FakeSearch(behaviour)
    monkeypatch.setattr(
        agent_mod, "duckduckgo_search_tool", lambda: type("T", (), {"function": fake})()
    )
    return fake


# --- happy path -----------------------------------------------------------


def test_results_pass_through(monkeypatch, sleeps):
    _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    assert asyncio.run(tool.function("q")) == [{"title": "hit"}]


# --- defect 1 & 6: exponential backoff, not linear -------------------------


def test_backoff_is_exponential_and_capped():
    delays = [agent_mod._backoff_delay(a, rate_limited=False) for a in range(1, 5)]
    # Strip jitter by checking each is within the jitter band of base * 2^(n-1).
    for attempt, delay in enumerate(delays, start=1):
        nominal = min(
            agent_mod._SEARCH_BACKOFF_BASE_S * (2 ** (attempt - 1)),
            agent_mod._SEARCH_BACKOFF_CAP_S,
        )
        assert nominal * 0.7 <= delay <= nominal * 1.3
    # Strictly growing until the cap, i.e. exponential not linear.
    assert delays[1] > delays[0] * 1.5
    assert delays[2] > delays[1] * 1.5


def test_backoff_never_exceeds_the_cap():
    worst = max(
        agent_mod._backoff_delay(a, rate_limited=True) for a in range(1, 12)
    )
    assert worst <= agent_mod._SEARCH_BACKOFF_CAP_S * (1 + agent_mod._SEARCH_JITTER)


# --- defect 5: a ratelimit waits longer than an ordinary error -------------


def test_ratelimit_backs_off_harder_than_a_plain_error():
    plain = [agent_mod._backoff_delay(2, rate_limited=False) for _ in range(40)]
    limited = [agent_mod._backoff_delay(2, rate_limited=True) for _ in range(40)]
    assert min(limited) > max(plain)


# --- defect 3: jitter --------------------------------------------------------


def test_backoff_is_jittered():
    values = {round(agent_mod._backoff_delay(2, rate_limited=False), 6) for _ in range(30)}
    assert len(values) > 1, "no jitter: retries would re-collide"


# --- defect 2: pacing state is global, not per-lead ------------------------


def test_pacing_state_is_shared_across_tool_instances(monkeypatch, sleeps):
    """
    A fresh research agent (and tool) is built per lead. If the clock were
    per-instance, lead N+1 would search immediately after lead N.
    """
    _install(monkeypatch, lambda n: [{"title": "hit"}])
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: 100.0)

    lead_one = agent_mod._paced_duckduckgo_tool(max_searches=4)
    asyncio.run(lead_one.function("q"))
    assert agent_mod._last_search_at == 100.0

    # A brand-new tool, as a new lead would get, must still observe the gap.
    sleeps.clear()
    lead_two = agent_mod._paced_duckduckgo_tool(max_searches=4)
    asyncio.run(lead_two.function("q"))
    assert sleeps and sleeps[0] == pytest.approx(agent_mod._SEARCH_MIN_INTERVAL_S), (
        "second lead did not wait: pacing state is not global"
    )


# --- defect 4: empty results are NOT reported as tool failures -------------


def test_genuine_empty_is_reported_as_absence_not_failure(monkeypatch, sleeps):
    """
    This is what keeps the company_footprint red flag working: a real
    'nothing out there' must not look like a tooling failure.
    """
    fake = _install(monkeypatch, lambda n: [])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("nonexistent holdings llc"))
    assert result.startswith("SEARCH_RETURNED_NO_RESULTS")
    assert "evidence of absence" in result
    # Retried once in case it was a soft throttle, then accepted.
    assert len(fake.calls) == 2


def test_soft_throttle_then_results_recovers(monkeypatch, sleeps):
    fake = _install(monkeypatch, lambda n: [] if n == 1 else [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    assert asyncio.run(tool.function("q")) == [{"title": "hit"}]
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "exc", [RatelimitException("429"), TimeoutException("slow"), DDGSException("boom"), RuntimeError("other")]
)
def test_errors_degrade_to_unavailable_never_raise(monkeypatch, sleeps, exc):
    def raiser(n):
        raise exc

    fake = _install(monkeypatch, raiser)
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    result = asyncio.run(tool.function("q"))
    assert result.startswith("SEARCH_UNAVAILABLE")
    assert "TOOLING FAILURE" in result
    assert "NOT evidence" in result
    assert len(fake.calls) == agent_mod._SEARCH_MAX_ATTEMPTS


def test_error_then_success_recovers(monkeypatch, sleeps):
    def flaky(n):
        if n < 3:
            raise RatelimitException("429")
        return [{"title": "hit"}]

    fake = _install(monkeypatch, flaky)
    tool = agent_mod._paced_duckduckgo_tool(max_searches=4)
    assert asyncio.run(tool.function("q")) == [{"title": "hit"}]
    assert len(fake.calls) == 3


# --- budget + serialisation ------------------------------------------------


def test_search_budget_is_enforced(monkeypatch, sleeps):
    fake = _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=2)

    async def run_all():
        return [await tool.function(f"q{i}") for i in range(4)]

    results = asyncio.run(run_all())
    assert len(fake.calls) == 2
    assert results[2].startswith("SEARCH_BUDGET_EXHAUSTED")


def test_concurrent_calls_are_serialised(monkeypatch, sleeps):
    fake = _install(monkeypatch, lambda n: [{"title": "hit"}])
    tool = agent_mod._paced_duckduckgo_tool(max_searches=8)

    async def run_parallel():
        await asyncio.gather(*(tool.function(f"q{i}") for i in range(5)))

    asyncio.run(run_parallel())
    assert fake.max_concurrent == 1, "concurrent searches are what trip the limiter"
