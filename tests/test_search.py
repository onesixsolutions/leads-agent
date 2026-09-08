"""
Tests for the research web-search tool.

Fully offline: engines are faked and sleeps captured rather than performed.
Each test pins a property that a real rate limit or outage would break.
"""

from __future__ import annotations

import asyncio

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from leads_agent import search as S

NO_RESULTS_EXC = DDGSException("No results found.")


class _FakeClient:
    """Stands in for ddgs.DDGS; `plan` maps backend -> behaviour."""

    def __init__(self, plan):
        self.plan = plan
        self.calls: list[tuple[str, str]] = []
        self.concurrent = 0
        self.max_concurrent = 0

    def text(self, query, *, max_results=None, backend=None):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append((backend, query))
        try:
            outcome = self.plan(backend, len(self.calls))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            self.concurrent -= 1


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float):
        recorded.append(seconds)

    monkeypatch.setattr(S.asyncio, "sleep", fake_sleep)
    return recorded


@pytest.fixture(autouse=True)
def _reset_clock(monkeypatch):
    monkeypatch.setattr(S, "_last_search_at", 0.0)


def _tool(monkeypatch, plan, *, budget=4, backends=("bing", "yahoo")) -> tuple:
    fake = _FakeClient(plan)
    monkeypatch.setattr(S, "DDGS", lambda **kw: fake)
    return S.web_search_tool(budget, backends=backends), fake


def _hit(title="hit"):
    return [{"title": title, "href": "https://x", "body": "b"}]


# --- the bug from the live log -------------------------------------------


def test_no_results_is_a_ddgs_exception_not_an_empty_list():
    """
    Regression guard: ddgs signals emptiness by RAISING. Earlier code assumed
    [] and so classified every genuine absence as a tooling failure.
    """
    assert S._is_no_results(NO_RESULTS_EXC)
    assert not S._is_no_results(TimeoutException("operation timed out"))
    assert not S._is_no_results(RatelimitException("429"))


def test_no_results_everywhere_reports_absence_not_failure(monkeypatch, sleeps):
    tool, fake = _tool(monkeypatch, lambda b, n: NO_RESULTS_EXC)
    result = asyncio.run(tool.function("nonexistent holdings llc"))
    assert result.startswith("SEARCH_RETURNED_NO_RESULTS")
    assert "evidence of absence" in result
    # Tried every engine once; did NOT burn retries on a deterministic answer.
    assert [b for b, _ in fake.calls] == ["bing", "yahoo"]


def test_no_results_is_not_retried(monkeypatch, sleeps):
    """Retrying a deterministic 'no results' wasted ~40s per query."""
    tool, fake = _tool(monkeypatch, lambda b, n: NO_RESULTS_EXC)
    asyncio.run(tool.function("q"))
    assert len(fake.calls) == 2, "should try each engine once, not retry"
    # One spacing sleep per engine call and nothing more; an extra sleep would
    # be a retry backoff on an answer that will never change.
    assert len(sleeps) <= len(fake.calls), (
        f"backed off on a deterministic no-results: {sleeps}"
    )


# --- failover ------------------------------------------------------------


def test_falls_over_to_the_next_engine(monkeypatch, sleeps):
    """The live failure: duckduckgo times out, another engine works."""
    def plan(backend, n):
        if backend == "duckduckgo":
            return TimeoutException("operation timed out")
        return _hit("from bing")

    tool, fake = _tool(monkeypatch, plan, backends=("duckduckgo", "bing"))
    result = asyncio.run(tool.function("Uline annual revenue"))
    assert result == [{"title": "from bing", "href": "https://x", "body": "b"}]
    assert [b for b, _ in fake.calls] == ["duckduckgo", "bing"]


def test_all_engines_failing_reports_unavailable(monkeypatch, sleeps):
    tool, fake = _tool(monkeypatch, lambda b, n: TimeoutException("timed out"))
    result = asyncio.run(tool.function("q"))
    assert result.startswith("SEARCH_UNAVAILABLE")
    assert "TOOLING FAILURE" in result and "NOT evidence" in result
    # Every engine, on every retry pass.
    assert len(fake.calls) == 2 * S.MAX_ATTEMPTS


def test_one_engine_empty_beats_others_failing(monkeypatch, sleeps):
    """
    If one engine genuinely ran and found nothing, that is an answer — do not
    report a tooling failure just because the others errored.
    """
    def plan(backend, n):
        return NO_RESULTS_EXC if backend == "bing" else TimeoutException("t")

    tool, _ = _tool(monkeypatch, plan)
    assert asyncio.run(tool.function("q")).startswith("SEARCH_RETURNED_NO_RESULTS")


def test_transient_failure_recovers_on_retry(monkeypatch, sleeps):
    def plan(backend, n):
        return _hit() if n > 2 else RatelimitException("429")

    tool, fake = _tool(monkeypatch, plan)
    assert asyncio.run(tool.function("q")) == _hit()
    assert len(fake.calls) == 3


# --- backoff -------------------------------------------------------------


def test_backoff_is_exponential_not_linear():
    delays = [S.backoff_delay(a, rate_limited=False) for a in range(1, 4)]
    for attempt, delay in enumerate(delays, start=1):
        nominal = min(S.BACKOFF_BASE_S * (2 ** (attempt - 1)), S.BACKOFF_CAP_S)
        assert nominal * 0.7 <= delay <= nominal * 1.3
    assert delays[1] > delays[0] * 1.5


def test_backoff_capped_and_ratelimit_weighted():
    assert max(S.backoff_delay(a, rate_limited=True) for a in range(1, 12)) <= (
        S.BACKOFF_CAP_S * (1 + S.JITTER)
    )
    assert min(S.backoff_delay(2, rate_limited=True) for _ in range(40)) > max(
        S.backoff_delay(2, rate_limited=False) for _ in range(40)
    )


def test_backoff_is_jittered():
    assert len({round(S.backoff_delay(2, rate_limited=False), 6) for _ in range(30)}) > 1


# --- pacing, budget, serialisation ---------------------------------------


def test_pacing_state_is_global_across_tool_instances(monkeypatch, sleeps):
    """A fresh tool is built per lead; the clock must not reset with it."""
    monkeypatch.setattr(S.time, "monotonic", lambda: 100.0)
    tool_a, _ = _tool(monkeypatch, lambda b, n: _hit())
    asyncio.run(tool_a.function("q"))
    assert S._last_search_at == 100.0

    sleeps.clear()
    tool_b, _ = _tool(monkeypatch, lambda b, n: _hit())
    asyncio.run(tool_b.function("q"))
    assert sleeps and sleeps[0] == pytest.approx(S.MIN_INTERVAL_S), (
        "second lead did not wait: pacing is not global"
    )


def test_budget_is_enforced(monkeypatch, sleeps):
    tool, fake = _tool(monkeypatch, lambda b, n: _hit(), budget=2)

    async def run_all():
        return [await tool.function(f"q{i}") for i in range(4)]

    results = asyncio.run(run_all())
    assert results[2].startswith("SEARCH_BUDGET_EXHAUSTED")
    assert len({q for _, q in fake.calls}) == 2


def test_calls_are_serialised(monkeypatch, sleeps):
    tool, fake = _tool(monkeypatch, lambda b, n: _hit(), budget=8)

    async def run_parallel():
        await asyncio.gather(*(tool.function(f"q{i}") for i in range(5)))

    asyncio.run(run_parallel())
    assert fake.max_concurrent == 1


def test_results_are_normalised(monkeypatch, sleeps):
    tool, _ = _tool(monkeypatch, lambda b, n: [{"title": "T", "url": "U", "description": "D"}])
    assert asyncio.run(tool.function("q")) == [{"title": "T", "href": "U", "body": "D"}]
