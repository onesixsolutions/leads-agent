"""
Tests for HubSpot lead parsing.

Regression cover for the empty-field bug found during ICP backtesting: an
empty `*Company*:` line used to swallow the following line, loading the
message text into `company` and sending research after a bogus entity.
"""

from __future__ import annotations

from leads_agent.models import HubSpotLead


def _fallback(**fields: str) -> str:
    labels = {
        "first_name": "First Name",
        "last_name": "Last Name",
        "email": "Email",
        "company": "Company",
        "message": "Message",
    }
    return "\n".join(f"*{labels[k]}*: {v}" for k, v in fields.items())


def test_parses_all_fields():
    lead = HubSpotLead._parse_hubspot_text(
        _fallback(
            first_name="Dana",
            last_name="Whitfield",
            email="<mailto:dana@example.com|dana@example.com>",
            company="Example Manufacturing",
            message="We need help migrating to Snowflake.",
        )
    )
    assert lead.first_name == "Dana"
    assert lead.last_name == "Whitfield"
    assert lead.email == "dana@example.com"
    assert lead.company == "Example Manufacturing"
    assert lead.message == "We need help migrating to Snowflake."


def test_empty_company_does_not_swallow_the_message():
    lead = HubSpotLead._parse_hubspot_text(
        _fallback(
            first_name="Jordan",
            last_name="Kim",
            email="jordan@gmail.com",
            company="",
            message="Working on a stealth AI product, pre-seed.",
        )
    )
    assert lead.company is None
    assert lead.message == "Working on a stealth AI product, pre-seed."


def test_empty_name_fields_stay_none():
    lead = HubSpotLead._parse_hubspot_text(
        _fallback(first_name="", last_name="", email="x@y.com", company="Acme", message="Hello")
    )
    assert lead.first_name is None
    assert lead.last_name is None
    assert lead.company == "Acme"


def test_multiline_message_is_preserved():
    text = (
        "*First Name*: Dana\n"
        "*Company*: Example Co\n"
        "*Message*: Line one.\nLine two.\nLine three."
    )
    lead = HubSpotLead._parse_hubspot_text(text)
    assert lead.company == "Example Co"
    assert lead.message == "Line one.\nLine two.\nLine three."


def test_field_order_does_not_matter():
    text = (
        "*Message*: We are consolidating warehouses.\n"
        "*Company*: Reordered Inc\n"
        "*Email*: a@b.com"
    )
    lead = HubSpotLead._parse_hubspot_text(text)
    # The message is multi-line and greedy, so it runs to the end of the text;
    # what matters is that the single-line fields are still found correctly.
    assert lead.company == "Reordered Inc"
    assert lead.email == "a@b.com"


def test_non_hubspot_event_returns_none():
    assert HubSpotLead.from_slack_event({"subtype": "bot_message", "username": "other"}) is None
    assert HubSpotLead.from_slack_event({"username": "hubspot"}) is None
    assert HubSpotLead.from_slack_event({"subtype": "bot_message", "username": "hubspot", "attachments": []}) is None
