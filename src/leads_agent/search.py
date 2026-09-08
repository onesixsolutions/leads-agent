"""
Web search for the research stage.

Backed by `ddgs`, which despite its name fronts several engines (bing, yahoo,
google, brave, mojeek, startpage, duckduckgo). The `duckduckgo` backend itself
is unreliable from a server IP — it rate-limits to the point of timing out on
every query — so this module treats the backend as a failover chain rather
than trusting any single engine.

Three guarantees for the caller:

1. Search never raises. Every failure becomes a message the model can reason
   about, because an exception here would discard the whole research stage.
2. "The search ran and found nothing" is reported differently from "the search
   never ran". The ICP `company_footprint` rule depends on that distinction:
   the first is evidence of absence, the second is not evidence at all.
3. Calls are serialised and spaced process-wide, and the per-lead budget is
   enforced for real rather than being a suggestion in the prompt.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import anyio.to_thread
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic_ai.tools import Tool

logger = logging.getLogger(__name__)

# Engines are tried in this order. bing and yahoo are the reliable pair from a
# server IP; the rest are kept as further fallbacks because availability moves
# around. `duckduckgo` is last on purpose — it is the one that blocks us.
DEFAULT_BACKENDS = ("bing", "yahoo", "google", "brave", "mojeek", "duckduckgo")

# ddgs defaults to a 5s timeout, which is where the "operation timed out"
# failures came from; the engines routinely take longer than that.
DEFAULT_TIMEOUT_S = 20

# Pacing. This state is module-level, NOT per-tool: a fresh research agent (and
# therefore a fresh tool) is built for every lead, so per-instance state would
# reset between leads and the first search of lead N+1 would fire immediately
# after the last search of lead N — exactly the burst that gets us throttled.
MIN_INTERVAL_S = 1.5
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0
RATELIMIT_FACTOR = 3.0
JITTER = 0.25

# A bare float rather than an asyncio primitive on purpose: each
# Agent.run_sync() may run on its own event loop, and an asyncio.Lock shared
# across loops raises. Intra-run serialisation uses a per-instance lock, which
# is always on a single loop; this only enforces the minimum gap, where a rare
# lost update costs one slightly short sleep.
_last_search_at: float = 0.0

# `ddgs` signals "nothing found" by raising rather than returning []. It is a
# plain DDGSException whose message is this string, so it has to be matched on
# text — but it means something completely different from a transport error and
# must never be retried as though it were one.
_NO_RESULTS_MARKER = "no results found"

BUDGET_SPENT = (
    "SEARCH_BUDGET_EXHAUSTED: no searches remain for this lead. Report what you "
    "have and mark anything still unestablished as unknown."
)
NO_RESULTS = (
    "SEARCH_RETURNED_NO_RESULTS: the search ran successfully across every engine "
    "and genuinely returned zero results for this query. This IS evidence of "
    "absence for this query — if well-formed searches for the company name "
    "and its domain all come back like this, that supports "
    "company_footprint = not_met."
)
UNAVAILABLE = (
    "SEARCH_UNAVAILABLE: the search tool errored or was rate-limited on every "
    "engine and did not run. This is a TOOLING FAILURE and is NOT evidence about "
    "the company — never treat it as an absence of web presence. Mark the "
    "affected criteria unknown and say the research was unavailable."
)


def _is_no_results(exc: Exception) -> bool:
    """True when ddgs is reporting an empty result set rather than a failure."""
    return isinstance(exc, DDGSException) and _NO_RESULTS_MARKER in str(exc).lower()


def backoff_delay(attempt: int, *, rate_limited: bool) -> float:
    """Exponential backoff with jitter, capped; longer when rate-limited."""
    delay = BACKOFF_BASE_S * (2 ** (attempt - 1))
    if rate_limited:
        delay *= RATELIMIT_FACTOR
    delay = min(delay, BACKOFF_CAP_S)
    return delay * (1.0 + random.uniform(-JITTER, JITTER))


async def _await_search_slot() -> None:
    """Sleep until the process-global minimum gap since the last search."""
    gap = MIN_INTERVAL_S - (time.monotonic() - _last_search_at)
    if gap > 0:
        await asyncio.sleep(gap)


def _normalise(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Reduce engine rows to the three fields the model actually uses."""
    out: list[dict[str, str]] = []
    for row in results:
        out.append(
            {
                "title": str(row.get("title", "")),
                "href": str(row.get("href") or row.get("url") or ""),
                "body": str(row.get("body") or row.get("description") or ""),
            }
        )
    return out


def web_search_tool(
    max_searches: int,
    *,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_results: int = 8,
) -> Tool:
    """
    Build the research agent's search tool.

    Args:
        max_searches: hard cap on searches for this lead.
        backends: engines to try in order before giving up on a query.
        timeout_s: per-request timeout handed to ddgs.
        max_results: rows requested per engine.
    """
    client = DDGS(timeout=timeout_s)
    lock = asyncio.Lock()
    state = {"calls": 0}

    async def _one_engine(query: str, backend: str) -> tuple[list[dict[str, str]] | None, bool, bool]:
        """
        Run one engine.

        Returns (results, empty, rate_limited). `results` is None when the
        engine failed; `empty` distinguishes a successful search that found
        nothing from a failure.
        """
        global _last_search_at
        try:
            raw = await anyio.to_thread.run_sync(
                lambda: client.text(query, max_results=max_results, backend=backend)
            )
        except RatelimitException as exc:
            logger.warning("search rate-limited on %s for %r: %s", backend, query, exc)
            return None, False, True
        # Deliberately broad: any failure here must degrade to a message
        # rather than escape and discard the whole research stage.
        except Exception as exc:  # noqa: BLE001
            if _is_no_results(exc):
                logger.info("search: no results on %s for %r", backend, query)
                return None, True, False
            level = logger.info if isinstance(exc, TimeoutException) else logger.warning
            level("search failed on %s for %r: %s", backend, query, exc)
            return None, False, False
        finally:
            _last_search_at = time.monotonic()

        if not raw:
            return None, True, False
        return _normalise(raw), False, False

    async def web_search(query: str) -> Any:
        """Searches the web for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        # Serialising matters: concurrent calls are what trip rate limiters.
        async with lock:
            if state["calls"] >= max_searches:
                return BUDGET_SPENT
            state["calls"] += 1

            any_engine_ran = False

            for attempt in range(1, MAX_ATTEMPTS + 1):
                rate_limited = False

                for backend in backends:
                    await _await_search_slot()
                    results, empty, limited = await _one_engine(query, backend)

                    if results:
                        return results
                    if empty:
                        # This engine worked and found nothing. Keep trying the
                        # others before concluding the web has nothing.
                        any_engine_ran = True
                    if limited:
                        rate_limited = True

                # Every engine either failed or found nothing on this pass. If
                # at least one genuinely ran, the query has no answer and
                # retrying will not change that.
                if any_engine_ran:
                    return NO_RESULTS

                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(backoff_delay(attempt, rate_limited=rate_limited))

            logger.warning("search exhausted every engine and retry for %r", query)
            return UNAVAILABLE

    return Tool(web_search, name="web_search", takes_ctx=False)
