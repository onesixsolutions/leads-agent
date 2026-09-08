"""
HTML rendering of a lead brief.

The Slack card answers "do I care?"; this page answers "why, and what do I do
about it?". It is the same content the console renderer in
`core.icp_report` shows, laid out for reading rather than for a terminal.

Everything is inlined — CSS in a `<style>` block, no scripts, no fonts, no
images. A brief is a static object in S3 that must render identically whether
it is served by the app, opened from a `file://` copy, or saved as a PDF, so
it cannot depend on anything the page does not carry itself.

The criteria list is driven by `icp_fit.CRITERIA`. There is deliberately no
second copy of the criterion names here: adding a criterion must change this
page automatically, never silently omit a row.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from leads_agent.icp_fit import CRITERIA, verdict_display
from leads_agent.models import CriterionStatus, ICPVerdict

# Status chip text. The mapping is display-only; the verdict logic lives in
# `icp_fit` and is never re-derived here.
_STATUS_TEXT: dict[CriterionStatus, str] = {
    CriterionStatus.met: "met",
    CriterionStatus.not_met: "not met",
    CriterionStatus.partial: "partial",
    CriterionStatus.unknown: "unknown",
}

_STYLE = """
:root {
  --bg: #f4f5f7;
  --surface: #ffffff;
  --border: #e2e5ea;
  --border-strong: #cdd2da;
  --ink: #14171c;
  --ink-soft: #4b5361;
  --ink-faint: #7b8494;
  --accent: #2f5bd8;
  --met: #14804a;
  --met-bg: #e6f4ec;
  --not-met: #b42318;
  --not-met-bg: #fbeae8;
  --partial: #97650b;
  --partial-bg: #fbf2e0;
  --unknown: #5c6470;
  --unknown-bg: #edeff2;
  --radius: 10px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #11141a;
    --surface: #191d25;
    --border: #2a303b;
    --border-strong: #39414f;
    --ink: #e8ebf0;
    --ink-soft: #b0b8c6;
    --ink-faint: #838d9d;
    --accent: #7fa0ff;
    --met: #5dd39b;
    --met-bg: #10281d;
    --not-met: #f79289;
    --not-met-bg: #2c1614;
    --partial: #e8be6a;
    --partial-bg: #2a2113;
    --unknown: #98a1b0;
    --unknown-bg: #22272f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 16px 72px;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-text-size-adjust: 100%;
}
.page { max-width: 760px; margin: 0 auto; }
a { color: var(--accent); }

.masthead { padding: 32px 0 20px; }
.eyebrow {
  margin: 0 0 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.masthead h1 { margin: 0; font-size: 27px; line-height: 1.2; letter-spacing: -0.015em; }
.masthead .sub { margin: 6px 0 0; color: var(--ink-soft); font-size: 15px; }

.verdict {
  border: 1px solid var(--border);
  border-left: 5px solid var(--verdict-color, var(--unknown));
  background: var(--surface);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 18px;
}
.verdict--in_icp { --verdict-color: var(--met); }
.verdict--partial_fit { --verdict-color: var(--partial); }
.verdict--needs_verification { --verdict-color: var(--accent); }
.verdict--out_of_icp { --verdict-color: var(--not-met); }
.verdict--not_evaluated { --verdict-color: var(--unknown); }
.verdict__label {
  margin: 0;
  font-size: 19px;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: var(--verdict-color);
}
.verdict__statement { margin: 10px 0 0; font-size: 16px; color: var(--ink); }
.verdict__reason { margin: 8px 0 0; color: var(--ink-soft); font-size: 14px; }
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 26px;
  margin: 16px 0 0;
  padding: 14px 0 0;
  border-top: 1px solid var(--border);
}
.meta div { min-width: 0; }
.meta dt {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.meta dd { margin: 2px 0 0; font-size: 14px; font-weight: 600; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  margin-bottom: 14px;
}
.card > h2 {
  margin: 0 0 14px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.card > h2 + p, .card > h2 + ul, .card > h2 + .criterion { margin-top: 0; }
.card p { margin: 0 0 10px; }
.card p:last-child { margin-bottom: 0; }

.criterion { padding: 12px 0; border-top: 1px solid var(--border); }
.criterion:first-of-type { border-top: 0; padding-top: 0; }
.criterion__head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; }
.criterion__name { font-weight: 650; font-size: 15px; }
.chip {
  display: inline-block;
  flex: none;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}
.chip--met { color: var(--met); background: var(--met-bg); }
.chip--not_met { color: var(--not-met); background: var(--not-met-bg); }
.chip--partial { color: var(--partial); background: var(--partial-bg); }
.chip--unknown { color: var(--unknown); background: var(--unknown-bg); }
.gate {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  padding: 1px 5px;
}
.criterion__finding { margin: 6px 0 0; font-size: 15px; color: var(--ink); }
.criterion__evidence {
  margin: 4px 0 0;
  font-size: 13px;
  color: var(--ink-soft);
  padding-left: 11px;
  border-left: 2px solid var(--border-strong);
}
.criterion__evidence b {
  font-weight: 700;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

ul.list { margin: 0; padding-left: 20px; }
ul.list li { margin: 0 0 7px; }
ul.list li:last-child { margin-bottom: 0; }

.card--in { border-left: 4px solid var(--met); }
.card--out { border-left: 4px solid var(--not-met); }
.card--open { border-left: 4px solid var(--partial); }

.note { margin: 0 0 14px; }
.note:last-child { margin-bottom: 0; }
.note__label {
  margin: 0 0 3px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.note__body { margin: 0; }

.facts { margin: 0; }
.facts div { display: flex; gap: 12px; padding: 7px 0; border-top: 1px solid var(--border); }
.facts div:first-child { border-top: 0; padding-top: 0; }
.facts dt { flex: 0 0 116px; color: var(--ink-faint); font-size: 13px; padding-top: 1px; }
.facts dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }

.quote {
  margin: 0;
  padding: 12px 14px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--ink-soft);
  font-size: 14px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.history { width: 100%; border-collapse: collapse; font-size: 14px; }
.history th {
  text-align: left;
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-faint);
  padding: 0 10px 8px 0;
  border-bottom: 1px solid var(--border);
}
.history td { padding: 10px 10px 10px 0; border-bottom: 1px solid var(--border); vertical-align: top; }
.history tr:last-child td { border-bottom: 0; }
.history .current { font-weight: 700; }

.footer {
  margin-top: 22px;
  color: var(--ink-faint);
  font-size: 12.5px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
}

@media (max-width: 520px) {
  .masthead h1 { font-size: 23px; }
  .facts div { flex-direction: column; gap: 2px; }
  .facts dt { flex: none; }
}
@media print {
  body { background: #fff; padding: 0; }
  .card, .verdict { border-color: #ccc; break-inside: avoid; }
}
"""


def _esc(value: Any) -> str:
    """HTML-escape any value; `None` becomes an empty string."""
    return html.escape(str(value), quote=True) if value is not None else ""


def _paragraphs(text: str) -> str:
    """Render free text, honouring blank-line paragraph breaks."""
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    return "".join(f"<p>{_esc(b).replace(chr(10), '<br>')}</p>" for b in blocks)


def _list(items: list[str]) -> str:
    return '<ul class="list">' + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>"


def _note(label: str, body: str) -> str:
    return (
        f'<div class="note"><p class="note__label">{_esc(label)}</p>'
        f'<p class="note__body">{_esc(body)}</p></div>'
    )


def _card(title: str, body: str, *, modifier: str = "") -> str:
    if not body:
        return ""
    cls = f"card {modifier}".strip()
    return f'<section class="{cls}"><h2>{_esc(title)}</h2>{body}</section>'


def _document(title: str, body: str) -> str:
    """Wrap rendered body markup in the standalone page shell."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        # These pages name individuals and judge them. Even behind a private
        # network, they should never end up in a search index.
        '<meta name="robots" content="noindex, nofollow, noarchive">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        f'<main class="page">{body}</main>\n'
        "</body>\n</html>\n"
    )


def _display_name(lead: Any, classification: Any) -> str:
    """Best available company name, falling back to the contact, then the id."""
    research = getattr(classification, "company_research", None)
    for candidate in (
        getattr(research, "company_name", None),
        getattr(classification, "company", None),
        getattr(lead, "company", None),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    contact = _contact_name(lead, classification)
    return contact or "Unidentified lead"


def _contact_name(lead: Any, classification: Any) -> str:
    research = getattr(classification, "contact_research", None)
    full = getattr(research, "full_name", None)
    if full and str(full).strip():
        return str(full).strip()
    for source in (classification, lead):
        first = (getattr(source, "first_name", None) or "").strip()
        last = (getattr(source, "last_name", None) or "").strip()
        if first or last:
            return f"{first} {last}".strip()
    return ""


def _verdict_section(lead: Any, classification: Any, meta_rows: list[tuple[str, str]]) -> str:
    verdict: ICPVerdict | None = getattr(classification, "icp_verdict", None)
    emoji, label = verdict_display(verdict)
    slug = verdict.value if verdict is not None else "not_evaluated"

    parts = [f'<section class="verdict verdict--{_esc(slug)}">']
    parts.append(f'<p class="verdict__label">{_esc(emoji)} {_esc(label)}</p>')

    brief = getattr(classification, "brief", None)
    statement = getattr(brief, "icp_statement", None)
    if statement:
        parts.append(f'<p class="verdict__statement">{_esc(statement)}</p>')

    reason = getattr(classification, "reason", None)
    if reason:
        parts.append(f'<p class="verdict__reason">{_esc(reason)}</p>')

    if meta_rows:
        parts.append('<dl class="meta">')
        parts.extend(
            f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in meta_rows
        )
        parts.append("</dl>")

    parts.append("</section>")
    return "".join(parts)


def _criteria_section(assessment: Any) -> str:
    if assessment is None:
        return ""

    rows: list[str] = []
    for spec in CRITERIA:
        criterion = getattr(assessment, spec.field, None)
        if criterion is None:
            continue
        status = getattr(criterion, "status", CriterionStatus.unknown)
        chip = _STATUS_TEXT.get(status, "unknown")
        gate = '<span class="gate">gate</span>' if spec.is_gate else ""

        row = [
            '<div class="criterion"><div class="criterion__head">',
            f'<span class="chip chip--{_esc(status.value)}">{_esc(chip)}</span>',
            f'<span class="criterion__name">{_esc(spec.label)}</span>{gate}',
            "</div>",
        ]
        if getattr(criterion, "finding", None):
            row.append(f'<p class="criterion__finding">{_esc(criterion.finding)}</p>')
        if getattr(criterion, "evidence", None):
            row.append(
                '<p class="criterion__evidence"><b>Evidence</b><br>'
                f"{_esc(criterion.evidence)}</p>"
            )
        row.append("</div>")
        rows.append("".join(row))

    return _card("ICP criteria", "".join(rows))


def _judgement_section(brief: Any) -> str:
    if brief is None:
        return ""
    notes: list[str] = []
    if getattr(brief, "analyst_take", None):
        notes.append(_note("Analyst take", brief.analyst_take))
    if getattr(brief, "opportunity", None):
        notes.append(_note("Opportunity", brief.opportunity))
    if getattr(brief, "exception_case", None):
        notes.append(_note("Exception case", brief.exception_case))
    if getattr(brief, "risks", None):
        notes.append(
            '<div class="note"><p class="note__label">Risks</p>'
            + _list(list(brief.risks))
            + "</div>"
        )
    return _card("Judgement", "".join(notes))


def _play_section(brief: Any) -> str:
    if brief is None:
        return ""
    parts: list[str] = []
    if getattr(brief, "recommended_entry", None):
        parts.append(_note("Recommended entry", brief.recommended_entry))
    if getattr(brief, "accelerators", None):
        parts.append(
            '<div class="note"><p class="note__label">Accelerators</p>'
            + _list(list(brief.accelerators))
            + "</div>"
        )
    if getattr(brief, "talking_points", None):
        parts.append(
            '<div class="note"><p class="note__label">Talking points</p>'
            + _list(list(brief.talking_points))
            + "</div>"
        )
    return _card("Recommended play", "".join(parts))


def _facts(rows: list[tuple[str, str]]) -> str:
    return (
        '<dl class="facts">'
        + "".join(f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in rows)
        + "</dl>"
    )


def _research_section(classification: Any) -> str:
    parts: list[str] = []

    company = getattr(classification, "company_research", None)
    if company is not None:
        rows = [("Company", getattr(company, "company_name", "") or "")]
        for label, value in (
            ("Industry", getattr(company, "industry", None)),
            ("Size", getattr(company, "company_size", None)),
            ("Website", getattr(company, "website", None)),
        ):
            if value:
                rows.append((label, value))
        body = _facts(rows)
        if getattr(company, "company_description", None):
            body += _paragraphs(company.company_description)
        if getattr(company, "relevance_notes", None):
            body += _note("Relevance", company.relevance_notes)
        parts.append(_card("Company research", body))

    contact = getattr(classification, "contact_research", None)
    if contact is not None:
        rows = [("Name", getattr(contact, "full_name", "") or "")]
        if getattr(contact, "title", None):
            rows.append(("Title", contact.title))
        body = _facts(rows)
        if getattr(contact, "linkedin_summary", None):
            body += _paragraphs(contact.linkedin_summary)
        if getattr(contact, "relevance_notes", None):
            body += _note("Relevance", contact.relevance_notes)
        parts.append(_card("Contact research", body))

    summary = getattr(classification, "research_summary", None)
    if summary:
        parts.append(_card("Research summary", _paragraphs(summary)))

    return "".join(parts)


def _source_section(lead: Any, classification: Any) -> str:
    rows: list[tuple[str, str]] = []
    contact = _contact_name(lead, classification)
    if contact:
        rows.append(("Contact", contact))
    email = getattr(classification, "email", None) or getattr(lead, "email", None)
    if email:
        rows.append(("Email", email))
    company = getattr(lead, "company", None)
    if company:
        rows.append(("Stated company", company))

    body = _facts(rows) if rows else ""

    summary = getattr(classification, "lead_summary", None)
    if summary:
        body += _note("Summary", summary)
    signals = getattr(classification, "key_signals", None)
    if signals:
        body += (
            '<div class="note"><p class="note__label">Signals</p>'
            + _list(list(signals))
            + "</div>"
        )
    message = getattr(lead, "message", None) or getattr(lead, "raw_text", None)
    if message:
        body += (
            '<div class="note"><p class="note__label">Their message</p>'
            f'<pre class="quote">{_esc(message)}</pre></div>'
        )

    return _card("Inbound lead", body)


def render_brief_html(
    lead: Any,
    classification: Any,
    *,
    version: int,
    generated_at: datetime | None = None,
    history_url: str | None = None,
) -> str:
    """
    Render the full lead brief as a standalone HTML document.

    Args:
        lead: The parsed `HubSpotLead`.
        classification: An `EnrichedLeadClassification` (a plain
            `LeadClassification` also renders — the ICP sections are simply
            omitted, which is what happens for spam).
        version: Version number shown in the header and footer.
        generated_at: Timestamp shown on the page; defaults to now (UTC).
        history_url: Link to this lead's version history, when serving.

    Returns:
        A complete `<!doctype html>` document with no external dependencies.
    """
    generated_at = generated_at or datetime.now(UTC)
    stamp = generated_at.strftime("%d %b %Y, %H:%M UTC")

    company_name = _display_name(lead, classification)
    contact = _contact_name(lead, classification)
    email = getattr(classification, "email", None) or getattr(lead, "email", None)
    subtitle = " · ".join(p for p in (contact, email) if p)

    label = getattr(classification, "label", None)
    action = getattr(classification, "action", None)
    meta_rows = [
        ("Triage", "GO — genuine inquiry" if getattr(label, "value", None) == "promising" else "Ignore"),
    ]
    if action is not None:
        meta_rows.append(("Action", action.value.replace("_", " ")))
    meta_rows.append(("Brief", f"v{version} · {stamp}"))

    brief = getattr(classification, "brief", None)

    body = [
        '<header class="masthead">',
        '<p class="eyebrow">OneSix · Lead brief</p>',
        f"<h1>{_esc(company_name)}</h1>",
        f'<p class="sub">{_esc(subtitle)}</p>' if subtitle else "",
        "</header>",
        _verdict_section(lead, classification, meta_rows),
        _card(
            "Why not in ICP",
            _list(list(classification.reasons_out_of_icp or [])),
            modifier="card--out",
        )
        if getattr(classification, "reasons_out_of_icp", None)
        else "",
        _card(
            "Why in ICP",
            _list(list(classification.reasons_in_icp or [])),
            modifier="card--in",
        )
        if getattr(classification, "reasons_in_icp", None)
        else "",
        _card(
            "Unverified — check before pursuing",
            _list(list(classification.open_questions or [])),
            modifier="card--open",
        )
        if getattr(classification, "open_questions", None)
        else "",
        _judgement_section(brief),
        _play_section(brief),
        _criteria_section(getattr(classification, "icp_assessment", None)),
        _research_section(classification),
        _source_section(lead, classification),
        '<p class="footer">'
        f"<span>Version {version} · generated {_esc(stamp)}</span>"
        + (f'<a href="{_esc(history_url)}">All versions</a>' if history_url else "")
        + "</p>",
    ]

    return _document(f"{company_name} — lead brief", "".join(p for p in body if p))


def _as_verdict(value: Any) -> ICPVerdict | None:
    """Coerce a stored verdict string back to the enum, tolerating junk."""
    try:
        return ICPVerdict(value)
    except ValueError:
        return None


def render_history_html(
    lead_id: str,
    versions: list[Any],
    *,
    current_version: int | None = None,
    brief_url_for: Any = None,
) -> str:
    """
    Render the version-history page for one lead.

    Args:
        lead_id: The lead this history belongs to.
        versions: `BriefVersion` entries, any order — sorted newest-first here.
        current_version: Version the bare `/briefs/<lead_id>` URL resolves to.
        brief_url_for: Callable taking a version number and returning its URL.
            When omitted, versions are listed without links.
    """
    ordered = sorted(versions, key=lambda v: getattr(v, "version", 0), reverse=True)
    title = ""
    for entry in ordered:
        title = getattr(entry, "company", None) or getattr(entry, "contact", None) or ""
        if title:
            break
    heading = title or lead_id

    rows: list[str] = []
    for entry in ordered:
        number = getattr(entry, "version", 0)
        is_current = number == current_version
        cell = f"v{number}"
        if brief_url_for is not None:
            cell = f'<a href="{_esc(brief_url_for(number))}">{_esc(cell)}</a>'
        emoji, verdict_label = verdict_display(_as_verdict(getattr(entry, "verdict", None)))
        rows.append(
            f'<tr><td class="{"current" if is_current else ""}">{cell}'
            + (" <small>(current)</small>" if is_current else "")
            + f"</td><td>{_esc(emoji)} {_esc(verdict_label)}</td>"
            + f"<td>{_esc(getattr(entry, 'created_at', '') or '')}</td>"
            + f"<td>{_esc(getattr(entry, 'action', '') or '')}</td></tr>"
        )

    table = (
        '<table class="history"><thead><tr><th>Version</th><th>Verdict</th>'
        "<th>Generated</th><th>Action</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        if rows
        else "<p>No briefs have been published for this lead yet.</p>"
    )

    body = (
        '<header class="masthead">'
        '<p class="eyebrow">OneSix · Lead brief history</p>'
        f"<h1>{_esc(heading)}</h1>"
        f'<p class="sub">{_esc(lead_id)}</p>'
        "</header>"
        + _card("Versions", table)
    )
    return _document(f"{heading} — brief history", body)
