"""
HTML rendering of a lead brief.

The Slack card answers "do I care?"; this page answers "why, and what do I say
on the call?".

Laid out for the actual reader: an AE with two minutes before a discovery call.
Everything needed to walk in prepared sits above the fold — verdict, the
one-sentence read, a gate strip you can scan in three seconds, what to say and
what to ask. The evidence that justifies all of it is one click away in
`<details>` sections rather than a wall of text you have to scroll past.

Everything is inlined — CSS and a few lines of progressive-enhancement JS, no
external fonts, images or scripts. A brief is a static object in S3 that must
render identically whether the app serves it, it is opened from a `file://`
copy, or it is printed, so it cannot depend on anything it does not carry.

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

# Status chip text. Display-only; the verdict logic lives in `icp_fit` and is
# never re-derived here.
_STATUS_TEXT: dict[CriterionStatus, str] = {
    CriterionStatus.met: "met",
    CriterionStatus.not_met: "not met",
    CriterionStatus.partial: "partial",
    CriterionStatus.unknown: "unverified",
}

# Short labels for the scan strip. The full labels are correct but too long to
# scan; these are only ever shown next to the full text elsewhere on the page.
_SHORT_LABEL: dict[str, str] = {
    "company_footprint": "Findable",
    "revenue_band": "Revenue band",
    "deal_shape": "Deal size",
    "platform": "Platform",
    "entry_door": "Door",
    "executive_sponsor": "Sponsor",
    "internal_team_depth": "Thin team",
    "trigger": "Trigger",
    "expansion_path": "Expansion",
    "buyer_persona": "Persona",
    "focus_overlay": "Overlay",
}

_STYLE = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  /* OneSix: near-black green ground, spring-green accent, violet secondary. */
  --ink:        #0a1410;
  --ink-2:      #12241b;
  --paper:      #f1f2ee;
  --surface:    #ffffff;
  --rule:       #dfe2da;
  --text:       #16211b;
  --muted:      #66716a;
  --faint:      #939b95;

  --spring:     #00d68f;
  --spring-dk:  #039e6b;
  --violet:     #7b53d8;

  --met:        #039e6b;
  --fail:       #cf3b30;
  --partial:    #b7791f;
  --unknown:    #8b95a1;

  --serif: "Iowan Old Style", "Charter", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: "Avenir Next", "Avenir", "Segoe UI", "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "SF Mono", "IBM Plex Mono", Menlo, "Cascadia Mono", monospace;

  --shadow: 0 1px 2px rgba(12,26,20,.05), 0 8px 24px -12px rgba(12,26,20,.18);
}

html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--text);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  /* Faint grain so the paper reads as a surface, not a void. */
  background-image: radial-gradient(rgba(10,20,16,.028) 1px, transparent 1px);
  background-size: 3px 3px;
}

.wrap { max-width: 940px; margin: 0 auto; padding: 0 20px 72px; }

/* --- sticky context bar: never lose the verdict while reading ----------- */
.bar {
  position: sticky; top: 0; z-index: 20;
  background: rgba(10,20,16,.94);
  backdrop-filter: saturate(140%) blur(8px);
  color: #eef2ee;
  border-bottom: 1px solid rgba(255,255,255,.10);
}
.bar__in {
  max-width: 940px; margin: 0 auto; padding: 9px 20px;
  display: flex; align-items: center; gap: 12px;
  font-size: 12.5px; letter-spacing: .01em;
}
.bar__co { font-weight: 600; }
.bar__sep { color: rgba(238,242,238,.34); }
.bar__meta { color: rgba(238,242,238,.62); }
.bar__spacer { margin-left: auto; }

/* --- hero --------------------------------------------------------------- */
.hero {
  background:
    radial-gradient(120% 140% at 88% -10%, rgba(0,214,143,.20), transparent 58%),
    radial-gradient(90% 120% at 8% 0%, rgba(123,83,216,.16), transparent 55%),
    var(--ink);
  color: #eef2ee;
  padding: 40px 0 34px;
  border-bottom: 3px solid var(--spring);
}
.hero__in { max-width: 940px; margin: 0 auto; padding: 0 20px; }
.eyebrow {
  margin: 0 0 16px;
  font-size: 10.5px; letter-spacing: .18em; text-transform: uppercase;
  color: rgba(238,242,238,.52);
  display: flex; align-items: center; gap: 9px;
}
.eyebrow::before {
  content: ""; width: 22px; height: 2px; background: var(--spring);
}
.hero h1 {
  margin: 0; font-family: var(--sans);
  font-size: clamp(26px, 4.4vw, 38px); font-weight: 600; letter-spacing: -.02em;
}
.hero__who {
  margin: 7px 0 0; color: rgba(238,242,238,.66); font-size: 14px;
}
.hero__who a { color: inherit; }

.verdict {
  display: inline-flex; align-items: center; gap: 9px;
  margin: 22px 0 0; padding: 7px 15px 7px 12px;
  border-radius: 999px; font-weight: 650; font-size: 13.5px;
  letter-spacing: .04em; text-transform: uppercase;
}
.verdict--in_icp        { background: var(--spring); color: #05211a; }
.verdict--out_of_icp    { background: #f0d7d4; color: #7d211a; }
.verdict--needs_verification { background: #f6e5c4; color: #6d4a12; }
.verdict--partial_fit   { background: #e5ddf7; color: #40287a; }
.verdict--not_evaluated { background: rgba(238,242,238,.16); color: #eef2ee; }

/* The single most important sentence on the page. */
.statement {
  margin: 24px 0 0; max-width: 62ch;
  font-family: var(--serif);
  font-size: clamp(18px, 2.5vw, 23px); line-height: 1.42;
  color: #f4f7f4; font-weight: 400;
}
.statement::first-letter { font-size: 1.02em; }

/* --- scan strip: the three-second layer --------------------------------- */
.scan {
  margin: -20px 0 0; position: relative; z-index: 5;
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 12px; box-shadow: var(--shadow); overflow: hidden;
}
.scan__hd {
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  padding: 13px 18px; border-bottom: 1px solid var(--rule);
  background: linear-gradient(180deg, #fbfcfa, var(--surface));
}
.scan__hd h2 {
  margin: 0; font-size: 10.5px; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); font-weight: 700;
}
.tally { font-size: 12.5px; color: var(--muted); font-family: var(--mono); }
.tally b { color: var(--text); font-weight: 650; }

.gates {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: 1px; background: var(--rule);
}
.gate {
  background: var(--surface); padding: 12px 14px 13px;
  display: flex; flex-direction: column; gap: 5px; min-width: 0;
}
.gate__k {
  font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--faint); font-weight: 650;
}
.gate__v {
  display: flex; align-items: center; gap: 7px;
  font-size: 13px; font-weight: 600;
}
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.s-met      { color: var(--met); }      .s-met .dot      { background: var(--met); }
.s-not_met  { color: var(--fail); }     .s-not_met .dot  { background: var(--fail); }
.s-partial  { color: var(--partial); }  .s-partial .dot  { background: var(--partial); }
.s-unknown  { color: var(--unknown); }  .s-unknown .dot  { background: var(--unknown); }
.gate--soft .gate__k::after {
  content: "·"; margin-left: 5px; color: var(--rule);
}

/* --- generic blocks ------------------------------------------------------ */
.grid { display: grid; gap: 16px; margin-top: 16px; }
@media (min-width: 760px) { .grid--2 { grid-template-columns: 1fr 1fr; } }

.block {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 12px; box-shadow: var(--shadow);
}
.block__hd {
  padding: 13px 18px 0;
  font-size: 10.5px; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); font-weight: 700;
}
.block__bd { padding: 10px 18px 16px; }
.block__bd > :first-child { margin-top: 0; }
.block__bd > :last-child { margin-bottom: 0; }

/* The analyst's read gets the most visual weight after the statement. */
.read {
  margin-top: 20px; padding: 18px 20px 18px 22px;
  background: var(--surface); border: 1px solid var(--rule);
  border-left: 3px solid var(--violet);
  border-radius: 10px; box-shadow: var(--shadow);
}
.read p { margin: 0 0 10px; font-size: 15.5px; line-height: 1.6; }
.read p:last-child { margin-bottom: 0; }

ol.plays, ul.asks { margin: 0; padding-left: 0; list-style: none; counter-reset: p; }
ol.plays li, ul.asks li {
  position: relative; padding-left: 26px; margin-bottom: 10px; line-height: 1.5;
}
ol.plays li:last-child, ul.asks li:last-child { margin-bottom: 0; }
ol.plays li { counter-increment: p; }
ol.plays li::before {
  content: counter(p); position: absolute; left: 0; top: 1px;
  width: 17px; height: 17px; border-radius: 50%;
  background: var(--ink); color: var(--spring);
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  display: grid; place-items: center;
}
ul.asks li::before {
  content: "?"; position: absolute; left: 0; top: 1px;
  width: 17px; height: 17px; border-radius: 50%;
  background: #f6e5c4; color: #6d4a12;
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  display: grid; place-items: center;
}

.pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0; list-style: none; }
.pills li {
  font-size: 12px; padding: 3px 9px; border-radius: 999px;
  background: #eef1ec; border: 1px solid var(--rule); color: var(--text);
}

.entry {
  font-family: var(--serif); font-size: 16px; line-height: 1.5;
  margin: 0 0 12px; padding: 12px 14px;
  background: #f3faf6; border: 1px solid #cdeadd; border-radius: 9px;
}

/* --- progressive disclosure --------------------------------------------- */
details.fold {
  margin-top: 12px; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 12px; box-shadow: var(--shadow);
}
details.fold > summary {
  list-style: none; cursor: pointer; user-select: none;
  padding: 14px 18px; display: flex; align-items: center; gap: 10px;
  font-size: 12px; letter-spacing: .1em; text-transform: uppercase;
  font-weight: 700; color: var(--text);
}
details.fold > summary::-webkit-details-marker { display: none; }
details.fold > summary::after {
  content: ""; margin-left: auto; width: 7px; height: 7px;
  border-right: 1.8px solid var(--muted); border-bottom: 1.8px solid var(--muted);
  transform: rotate(45deg) translateY(-2px); transition: transform .18s ease;
}
details.fold[open] > summary::after { transform: rotate(225deg) translateY(-2px); }
details.fold[open] > summary { border-bottom: 1px solid var(--rule); }
details.fold > summary:hover { background: #f8faf8; }
.fold__ct {
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  color: var(--muted); background: #eef1ec;
  border-radius: 999px; padding: 1px 8px;
}
.fold__bd { padding: 16px 18px 18px; }
.fold__bd > :first-child { margin-top: 0; }
.fold__bd > :last-child { margin-bottom: 0; }

/* --- criteria matrix ----------------------------------------------------- */
.crit { border-top: 1px solid var(--rule); padding: 12px 0; }
.crit:first-child { border-top: 0; padding-top: 0; }
.crit:last-child { padding-bottom: 0; }
.crit__hd { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.crit__nm { font-weight: 650; font-size: 14px; }
.chip {
  font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase;
  font-weight: 700; padding: 2px 8px; border-radius: 999px;
  border: 1px solid currentColor;
}
.gatetag {
  font-size: 10px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--faint); font-weight: 700;
}
.crit__find { margin: 6px 0 0; }
.crit__ev {
  margin: 7px 0 0; padding: 8px 11px;
  background: #f7f8f5; border-left: 2px solid var(--rule); border-radius: 0 6px 6px 0;
  font-family: var(--mono); font-size: 12px; line-height: 1.5; color: var(--muted);
  overflow-wrap: anywhere;
}
.crit__ev b { color: var(--text); font-weight: 650; }

/* --- misc ---------------------------------------------------------------- */
.why { margin: 0; padding-left: 18px; }
.why li { margin-bottom: 7px; }
.why li:last-child { margin-bottom: 0; }
.why--out li::marker { color: var(--fail); }
.why--in li::marker { color: var(--met); }
.why--open li::marker { color: var(--partial); }

.facts { display: grid; gap: 9px 18px; margin: 0 0 12px; }
@media (min-width: 560px) { .facts { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); } }
.facts dt {
  font-size: 10.5px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--faint); font-weight: 700;
}
.facts dd { margin: 1px 0 0; overflow-wrap: anywhere; }

.label {
  font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); font-weight: 700; margin: 16px 0 6px;
}
.label:first-child { margin-top: 0; }

.quote {
  margin: 0; padding: 13px 15px; background: #f7f8f5;
  border: 1px solid var(--rule); border-radius: 9px;
  font-family: var(--mono); font-size: 12.5px; line-height: 1.6;
  white-space: pre-wrap; overflow-wrap: anywhere;
}

.footer {
  margin: 30px 0 0; padding-top: 16px; border-top: 1px solid var(--rule);
  display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
  font-size: 12.5px; color: var(--faint);
}
.footer a { color: var(--spring-dk); font-weight: 600; }

.empty { color: var(--muted); font-style: italic; }

/* One orchestrated entrance rather than scattered micro-animations. */
@media (prefers-reduced-motion: no-preference) {
  .anim { animation: rise .5s cubic-bezier(.22,.68,.32,1) both; }
  .anim:nth-of-type(1) { animation-delay: .02s; }
  .anim:nth-of-type(2) { animation-delay: .07s; }
  .anim:nth-of-type(3) { animation-delay: .12s; }
  .anim:nth-of-type(4) { animation-delay: .16s; }
  .anim:nth-of-type(n+5) { animation-delay: .2s; }
  @keyframes rise { from { opacity: 0; transform: translateY(9px); } }
}

@media print {
  .bar { display: none; }
  body { background: #fff; }
  .hero { background: var(--ink) !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  details.fold { break-inside: avoid; }
  details.fold > summary::after { display: none; }
  .anim { animation: none !important; }
}
"""

# Progressive enhancement only: every section is readable with JS disabled,
# because a brief must survive being opened from a file:// copy or a PDF.
_SCRIPT = """
(function () {
  var toggle = document.getElementById('expand-all');
  if (!toggle) return;
  var folds = Array.prototype.slice.call(document.querySelectorAll('details.fold'));
  if (!folds.length) { toggle.hidden = true; return; }
  toggle.hidden = false;
  toggle.addEventListener('click', function () {
    var open = folds.some(function (f) { return !f.open; });
    folds.forEach(function (f) { f.open = open; });
    toggle.textContent = open ? 'Collapse all' : 'Expand all';
  });
})();
"""


# --- small helpers ---------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _paragraphs(text: str) -> str:
    blocks = [b.strip() for b in str(text).split("\n\n") if b.strip()]
    return "".join(f"<p>{_esc(b)}</p>" for b in blocks)


def _bullets(items: list[str], cls: str = "why") -> str:
    return f'<ul class="{cls}">' + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>"


def _facts(rows: list[tuple[str, str]]) -> str:
    return (
        '<dl class="facts">'
        + "".join(f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in rows)
        + "</dl>"
    )


def _block(title: str, body: str) -> str:
    if not body:
        return ""
    return (
        f'<section class="block anim"><p class="block__hd">{_esc(title)}</p>'
        f'<div class="block__bd">{body}</div></section>'
    )


def _fold(title: str, body: str, *, count: str = "", open_: bool = False) -> str:
    """A collapsed section. Counts on the summary let you judge before opening."""
    if not body:
        return ""
    badge = f'<span class="fold__ct">{_esc(count)}</span>' if count else ""
    attr = " open" if open_ else ""
    return (
        f'<details class="fold anim"{attr}><summary>{_esc(title)}{badge}</summary>'
        f'<div class="fold__bd">{body}</div></details>'
    )


def _document(title: str, body: str) -> str:
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
        f"<main>{body}</main>\n"
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )


def _display_name(lead: Any, classification: Any) -> str:
    for source in (
        getattr(getattr(classification, "company_research", None), "company_name", None),
        getattr(classification, "company", None),
        getattr(lead, "company", None),
    ):
        if source:
            return str(source)
    email = getattr(classification, "email", None) or getattr(lead, "email", None)
    if email and "@" in str(email):
        return str(email).split("@", 1)[1]
    return "Unidentified company"


def _contact_name(lead: Any, classification: Any) -> str:
    for first, last in (
        (getattr(classification, "first_name", None), getattr(classification, "last_name", None)),
        (getattr(lead, "first_name", None), getattr(lead, "last_name", None)),
    ):
        name = f"{first or ''} {last or ''}".strip()
        if name:
            return name
    full = getattr(getattr(classification, "contact_research", None), "full_name", None)
    return str(full) if full else ""


# --- sections --------------------------------------------------------------


def _scan_strip(assessment: Any) -> str:
    """
    The three-second layer: every gate, its status, and the running tally.

    Only gates appear. A non-gate cannot change the verdict, so putting one
    here would invite the reader to weigh something that does not count.
    """
    if assessment is None:
        return ""

    tiles: list[str] = []
    counts = {s: 0 for s in CriterionStatus}
    for spec in CRITERIA:
        if not spec.is_gate:
            continue
        criterion = getattr(assessment, spec.field, None)
        if criterion is None:
            continue
        counts[criterion.status] += 1
        label = _SHORT_LABEL.get(spec.field, spec.label)
        soft = "" if spec.hard else " gate--soft"
        tiles.append(
            f'<div class="gate{soft}">'
            f'<span class="gate__k">{_esc(label)}</span>'
            f'<span class="gate__v s-{criterion.status.value}">'
            f'<span class="dot"></span>{_esc(_STATUS_TEXT[criterion.status])}</span>'
            "</div>"
        )

    if not tiles:
        return ""

    total = sum(counts.values())
    bits = [f"<b>{total}</b> gates"]
    for status, word in (
        (CriterionStatus.not_met, "failed"),
        (CriterionStatus.unknown, "unverified"),
        (CriterionStatus.partial, "partial"),
        (CriterionStatus.met, "met"),
    ):
        if counts[status]:
            bits.append(f"<b>{counts[status]}</b> {word}")

    return (
        '<section class="scan anim">'
        '<div class="scan__hd"><h2>Gate check</h2>'
        f'<span class="tally">{" · ".join(bits)}</span></div>'
        f'<div class="gates">{"".join(tiles)}</div>'
        "</section>"
    )


def _read_section(brief: Any) -> str:
    """The analyst's judgement — the highest-value prose on the page."""
    take = getattr(brief, "analyst_take", None) if brief else None
    if not take:
        return ""
    return f'<section class="read anim">{_paragraphs(take)}</section>'


def _prep_section(brief: Any, classification: Any) -> str:
    """
    What to say and what to ask, side by side.

    This is the only part of the page that gets used *during* the call, so it
    sits above every piece of evidence rather than after it.
    """
    if brief is None:
        return ""

    plays = list(getattr(brief, "talking_points", None) or [])
    asks = list(getattr(classification, "open_questions", None) or [])

    left = ""
    if plays:
        left = _block(
            "What to say",
            '<ol class="plays">' + "".join(f"<li>{_esc(p)}</li>" for p in plays) + "</ol>",
        )
    right = ""
    if asks:
        right = _block(
            "What to ask",
            '<ul class="asks">' + "".join(f"<li>{_esc(a)}</li>" for a in asks) + "</ul>",
        )

    columns = f'<div class="grid grid--2">{left}{right}</div>' if (left and right) else (left or right)

    entry = getattr(brief, "recommended_entry", None)
    accelerators = list(getattr(brief, "accelerators", None) or [])
    tail = ""
    if entry or accelerators:
        body = f'<p class="entry">{_esc(entry)}</p>' if entry else ""
        if accelerators:
            body += '<p class="label">Accelerators</p><ul class="pills">' + "".join(
                f"<li>{_esc(a)}</li>" for a in accelerators
            ) + "</ul>"
        tail = _block("Recommended entry", body)

    return columns + tail


def _verdict_rationale(classification: Any) -> str:
    """Why the verdict came out the way it did, in the reader's own terms."""
    parts: list[str] = []
    for title, key, cls in (
        ("Why not in ICP", "reasons_out_of_icp", "why why--out"),
        ("Why in ICP", "reasons_in_icp", "why why--in"),
        ("Unverified — check before the call", "open_questions", "why why--open"),
    ):
        items = list(getattr(classification, key, None) or [])
        if items:
            parts.append(f'<p class="label">{title}</p>' + _bullets(items, cls))
    return "".join(parts)


def _criteria_detail(assessment: Any) -> tuple[str, int]:
    """Every criterion with its finding and the evidence behind it."""
    if assessment is None:
        return "", 0

    rows: list[str] = []
    for spec in CRITERIA:
        criterion = getattr(assessment, spec.field, None)
        if criterion is None:
            continue
        status = criterion.status
        tag = "gate" if spec.is_gate else "context"
        finding = getattr(criterion, "finding", "") or ""
        evidence = getattr(criterion, "evidence", None)
        rows.append(
            '<div class="crit">'
            f'<div class="crit__hd"><span class="crit__nm">{_esc(spec.label)}</span>'
            f'<span class="chip s-{status.value}">{_esc(_STATUS_TEXT[status])}</span>'
            f'<span class="gatetag">{tag}</span></div>'
            + (f'<p class="crit__find">{_esc(finding)}</p>' if finding else "")
            + (f'<p class="crit__ev"><b>Evidence</b> — {_esc(evidence)}</p>' if evidence else "")
            + "</div>"
        )
    return "".join(rows), len(rows)


def _risk_detail(brief: Any) -> str:
    if brief is None:
        return ""
    parts: list[str] = []
    for title, value in (
        ("Opportunity", getattr(brief, "opportunity", None)),
        ("Exception case", getattr(brief, "exception_case", None)),
    ):
        if value:
            parts.append(f'<p class="label">{title}</p>' + _paragraphs(value))
    risks = list(getattr(brief, "risks", None) or [])
    if risks:
        parts.append('<p class="label">Risks</p>' + _bullets(risks))
    return "".join(parts)


def _research_detail(classification: Any) -> str:
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
        parts.append('<p class="label">Company</p>' + _facts(rows))
        if getattr(company, "company_description", None):
            parts.append(_paragraphs(company.company_description))
        if getattr(company, "relevance_notes", None):
            parts.append(_paragraphs(company.relevance_notes))

    contact = getattr(classification, "contact_research", None)
    if contact is not None:
        rows = [("Name", getattr(contact, "full_name", "") or "")]
        if getattr(contact, "title", None):
            rows.append(("Title", contact.title))
        parts.append('<p class="label">Contact</p>' + _facts(rows))
        if getattr(contact, "linkedin_summary", None):
            parts.append(_paragraphs(contact.linkedin_summary))
        if getattr(contact, "relevance_notes", None):
            parts.append(_paragraphs(contact.relevance_notes))

    summary = getattr(classification, "research_summary", None)
    if summary:
        parts.append('<p class="label">Research summary</p>' + _paragraphs(summary))

    return "".join(parts)


def _inbound_detail(lead: Any, classification: Any) -> str:
    parts: list[str] = []

    rows: list[tuple[str, str]] = []
    email = getattr(classification, "email", None) or getattr(lead, "email", None)
    if email:
        rows.append(("Email", email))
    if getattr(lead, "company", None):
        rows.append(("Stated company", lead.company))
    if rows:
        parts.append(_facts(rows))

    summary = getattr(classification, "lead_summary", None)
    if summary:
        parts.append('<p class="label">Summary</p>' + _paragraphs(summary))

    signals = list(getattr(classification, "key_signals", None) or [])
    if signals:
        parts.append(
            '<p class="label">Signals</p><ul class="pills">'
            + "".join(f"<li>{_esc(s)}</li>" for s in signals)
            + "</ul>"
        )

    message = getattr(lead, "message", None) or getattr(lead, "raw_text", None)
    if message:
        parts.append(f'<p class="label">Their message</p><pre class="quote">{_esc(message)}</pre>')

    return "".join(parts)


# --- documents -------------------------------------------------------------


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
    who = " · ".join(p for p in (contact, str(email) if email else "") if p)

    verdict = getattr(classification, "icp_verdict", None)
    emoji, verdict_label = verdict_display(verdict)
    verdict_key = verdict.value if isinstance(verdict, ICPVerdict) else "not_evaluated"

    label = getattr(classification, "label", None)
    action = getattr(classification, "action", None)
    action_text = action.value.replace("_", " ") if action is not None else ""
    triage_text = "GO" if getattr(label, "value", None) == "promising" else "IGNORE"

    brief = getattr(classification, "brief", None)
    statement = getattr(brief, "icp_statement", None) if brief else None
    assessment = getattr(classification, "icp_assessment", None)

    criteria_html, criteria_count = _criteria_detail(assessment)

    bar_bits = [
        f'<span class="bar__co">{_esc(company_name)}</span>',
        f'<span class="bar__sep">/</span><span class="bar__meta">{_esc(verdict_label)}</span>',
    ]
    if action_text:
        bar_bits.append(
            f'<span class="bar__sep">/</span><span class="bar__meta">{_esc(action_text)}</span>'
        )
    bar_bits.append(
        '<span class="bar__spacer"></span>'
        f'<button id="expand-all" hidden class="fold__ct" type="button">Expand all</button>'
        f'<span class="bar__meta">v{version}</span>'
    )

    hero = (
        '<section class="hero"><div class="hero__in">'
        '<p class="eyebrow">OneSix · Lead brief</p>'
        f"<h1>{_esc(company_name)}</h1>"
        + (f'<p class="hero__who">{_esc(who)}</p>' if who else "")
        + f'<p class="verdict verdict--{verdict_key}">'
        f'<span aria-hidden="true">{emoji}</span>{_esc(verdict_label)}</p>'
        + (f'<p class="statement">{_esc(statement)}</p>' if statement else "")
        + "</div></section>"
    )

    body = [
        f'<div class="bar"><div class="bar__in">{"".join(bar_bits)}</div></div>',
        hero,
        '<div class="wrap">',
        _scan_strip(assessment),
        _read_section(brief),
        _prep_section(brief, classification),
        _fold("Why this verdict", _verdict_rationale(classification)),
        _fold("Criteria & evidence", criteria_html, count=str(criteria_count) if criteria_count else ""),
        _fold("Risks & opportunity", _risk_detail(brief)),
        _fold("Research & sources", _research_detail(classification)),
        _fold("The inbound lead", _inbound_detail(lead, classification)),
        '<p class="footer">'
        f"<span>Triage {_esc(triage_text)}"
        + (f" · action {_esc(action_text)}" if action_text else "")
        + f" · Version {version} · {_esc(stamp)}</span>"
        + (f'<a href="{_esc(history_url)}">All versions →</a>' if history_url else "")
        + "</p>",
        "</div>",
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

    Returns:
        A complete HTML document listing every stored version, newest first.
    """
    ordered = sorted(versions, key=lambda v: getattr(v, "version", 0), reverse=True)

    rows: list[str] = []
    for entry in ordered:
        number = getattr(entry, "version", 0)
        verdict = _as_verdict(getattr(entry, "verdict", None))
        emoji, verdict_label = verdict_display(verdict)
        key = verdict.value if verdict is not None else "not_evaluated"
        created = getattr(entry, "created_at", "") or ""
        current = " (current)" if current_version == number else ""
        href = brief_url_for(number) if callable(brief_url_for) else f"v{number}"
        rows.append(
            '<div class="crit">'
            f'<div class="crit__hd"><span class="crit__nm">'
            f'<a href="{_esc(href)}">Version {number}</a></span>'
            f'<span class="chip verdict--{key}" style="border-color:transparent">'
            f"{emoji} {_esc(verdict_label)}</span>"
            f'<span class="gatetag">{_esc(created)}{current}</span></div>'
            "</div>"
        )

    listing = "".join(rows) or '<p class="empty">No briefs have been published for this lead yet.</p>'

    body = (
        '<section class="hero"><div class="hero__in">'
        '<p class="eyebrow">OneSix · Brief history</p>'
        f"<h1>{_esc(lead_id)}</h1>"
        f'<p class="hero__who">{len(ordered)} version(s), newest first. '
        "Briefs are append-only, so every earlier assessment stays readable.</p>"
        "</div></section>"
        f'<div class="wrap">{_block("Versions", listing)}</div>'
    )
    return _document(f"{lead_id} — brief history", body)
