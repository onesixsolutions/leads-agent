"""
Tests for lead selection in `backtest` (--list / --index / --ts).

Offline: exercises extraction and selection only, never the LLM pipeline.
"""

from __future__ import annotations

import json

import pytest

from leads_agent.core.backtest import extract_leads_from_events, load_events_from_file


def _event(ts: str, first: str, email: str) -> dict:
    fb = f"*First Name*: {first}\n*Email*: <mailto:{email}|{email}>\n*Message*: Hello from {first}."
    return {
        "type": "events_api",
        "payload": {
            "event": {
                "type": "message",
                "subtype": "bot_message",
                "username": "HubSpot",
                "ts": ts,
                "attachments": [{"fallback": fb}],
            }
        },
    }


@pytest.fixture
def events() -> list[dict]:
    return [
        _event("1000.0001", "Alice", "alice@a.com"),
        {"type": "message", "text": "a human message, not a lead"},
        _event("1000.0002", "Bob", "bob@b.com"),
        _event("1000.0003", "Cara", "cara@c.com"),
    ]


def test_extraction_skips_non_hubspot_messages(events):
    leads = list(extract_leads_from_events(events))
    assert [l.first_name for _, l in leads] == ["Alice", "Bob", "Cara"]


def test_index_is_position_among_leads_not_messages(events):
    """
    The index --list prints (and --index selects) must count leads, not raw
    messages — otherwise the interleaved human message would shift it.
    """
    leads = list(extract_leads_from_events(events))
    by_index = {i: lead for i, (_, lead) in enumerate(leads, start=1)}
    assert by_index[2].first_name == "Bob"
    assert by_index[2].email == "bob@b.com"


def test_timestamp_selects_the_same_lead_regardless_of_position(events):
    """--ts must be stable even when newer leads shift every index."""
    leads = list(extract_leads_from_events(events))
    picked = [lead for ev, lead in leads if ev["ts"] == "1000.0003"]
    assert len(picked) == 1 and picked[0].first_name == "Cara"

    # Prepend a newer lead: Cara's index changes, her ts does not.
    shifted = [_event("1000.0009", "Dave", "dave@d.com"), *events]
    shifted_leads = list(extract_leads_from_events(shifted))
    assert shifted_leads[0][1].first_name == "Dave"
    still = [lead for ev, lead in shifted_leads if ev["ts"] == "1000.0003"]
    assert len(still) == 1 and still[0].first_name == "Cara"


def test_load_events_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="Expected JSON array"):
        load_events_from_file(p)


def test_load_events_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_events_from_file(tmp_path / "nope.json")
