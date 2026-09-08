# Leads Agent roadmap — operationalizing ICP²

Derived from `docs/ICP Squared 2026 v2.pdf` (working draft v2, July 2026).

## Why this roadmap exists

The deployed `prompt_config.json` now carries the ICP² doctrine, but it carries it as **prose in
free-text prompt fields**. That is the ceiling we are hitting:

- The agent cannot *enforce* the rules that ICP² states as hard rules (the $250M–$10B band, the
  $150k new-logo floor and its expansion exemption). An LLM reading a paragraph will apply them
  probabilistically.
- `PromptConfig` (`src/leads_agent/prompts/manager.py`) has no place to put a revenue band, a door,
  a persona, a platform tier or a floor. Everything nuanced has to be smuggled into
  `custom_instructions`.
- `build_research_prompt()` reuses the same list fields to generate DuckDuckGo operator clauses, so
  descriptive ICP entries produce unusable search queries. Doctrine and query tokens are coupled.
- Research is a single DuckDuckGo agent with a *suggested* search cap. It cannot reliably answer the
  one question that gates everything — what is this company's revenue?
- HubSpot is only an upstream Slack bot message parsed by regex (`HubSpotLead._parse_hubspot_text`).
  We cannot tell a net-new logo from an existing client, which is exactly the distinction the floor
  turns on, and we write nothing back.
- ICP² says the teeth live in HubSpot: "ICP definition, client personas and the 14-point scorecard
  become required fields on every new opportunity in HubSpot. No score, no stage advance."

The roadmap below turns the doctrine into structure, then into research depth, then into HubSpot
plumbing, then into GTM output.

## Phases

| Phase | Theme | Issues |
|-------|-------|--------|
| 0 | Foundation — make ICP² a data model, not a paragraph | 1–4 |
| 1 | Deeper research — answer the gating questions with evidence | 5–8 |
| 2 | HubSpot integration — read context, write the scorecard | 9–12 |
| 3 | GTM enablement — prepare the outreach | 13–16 |

Phase 0 is a hard prerequisite: issues 2 and 3 define the output contract that phases 1–3 fill in
and consume. Phases 1 and 2 can run in parallel once Phase 0 lands.

---

## Phase 0 — Foundation

### Issue 1 — Model the ICP² framework as structured config

**Problem.** `ICPConfig` has six flat string lists. ICP² is a structured framework: a revenue band
subdivided into three selling bands, six doors across two wings, six personas, five focus overlays,
a graduated platform score, a deal-size floor with named exemptions, four triggers, and
disqualifiers that are distinct from caution flags. All of that currently lives in
`custom_instructions` prose. Separately, `build_research_prompt()` derives search-operator clauses
from the same lists it derives doctrine from, so the two uses fight each other.

**Scope.**
- Extend `src/leads_agent/prompts/manager.py` with typed models: `RevenueBand`
  (`min_usd`, `max_usd`, selling sub-bands with first-contact titles and a how-to-sell note),
  `Door` (id, name, wing, description, accelerators), `Persona` (name, buyer type, maturity,
  titles, doors, overlays, needs, objections, what-they-buy, how-to-message), `FocusOverlay`
  (name, type: vertical/horizontal/channel, use cases, accelerators, channel, proof points),
  `PlatformTier` (score 2/1/0 with matching signals), `DealShape` (`new_logo_floor_usd`,
  `floor_exemptions`, target duration, expansion requirement), `Trigger`, and
  `disqualifiers` vs `caution_flags` as separate fields.
- Add an explicit `search_terms` block (industry / role / geo / technographic / firmographic
  tokens) and make `build_research_prompt()`'s clause pack read **only** from it. Doctrine fields
  stop feeding query generation.
- Keep every new field optional and keep backwards compatibility with the current flat schema so
  the deployed config and `prompt_config.example.json` both keep loading.
- Ship a full ICP²-shaped `prompt_config.example.json` and update the schema docs in
  `docs/ARCHITECTURE.md`.

**Acceptance criteria.**
- A config expressing the full ICP² framework loads, and `leads-agent prompts --full` renders bands,
  doors, personas, overlays, platform tiers, floor and triggers in the prompts.
- The old flat config still loads with no error (regression).
- Clause-pack output contains only short query tokens — no doctrine sentences.

**Files.** `src/leads_agent/prompts/manager.py`, `prompts/prompts.py`, `prompts/utils.py`,
`prompt_config.example.json`, `docs/ARCHITECTURE.md`
**Depends on.** — · **Size.** L

---

### Issue 2 — Replace the flat 1–5 score with an ICP² scorecard

**Problem.** `EnrichedLeadClassification` emits `score: 1-5` plus a one-line `score_reason`. That
tells an AE nothing about *why* a lead qualifies or which attribute is missing. ICP² scores on named
dimensions and, in Part 2, on a 7-attribute / 14-point project scorecard with explicit bands
(11–14 ideal, 7–10 conditional, 0–6 anti-project).

**Scope.**
- Add `ICPFit` to `src/leads_agent/models.py`: `revenue_band_fit` (in_band / below / above / unknown
  + evidence), `conviction` (sponsor named?, budget substantiated?), `internal_team_depth`
  (thin / capable / unknown), `platform_tier` (2/1/0 + evidence), `door` (one of six + confidence),
  `persona` (one of six + confidence), `overlay` (or none), `triggers` (list of detected triggers),
  `entry_shape` (estimated size band, new logo vs expansion), `designed_expansion` (path named?).
- Add `ProjectScorecard`: the seven Part 2 attributes each scored 0/1/2 with a rationale, a derived
  total, and a band label.
- Derive the existing `score` (1–5) and `action` from the structured fit so downstream consumers and
  `format_slack_message` keep working; keep `score_reason` as the one-sentence ICP² statement
  ("$800M PE-backed manufacturer, new CDO, migrating to Snowflake — Modernization & Migration door,
  ~$175k foundation build").
- Add a `missing_attributes` list so a conditional lead says what to go find out.
- Update the scoring prompt (`BASE_SCORING_PROMPT`) and `format_slack_message` to render the
  scorecard compactly.

**Acceptance criteria.**
- `leads-agent classify "<message>"` emits a populated `ICPFit` with per-dimension evidence.
- Every dimension the research stage could not establish reads `unknown` rather than being guessed.
- Slack output shows the one-sentence statement, the door, the persona and any missing attributes.

**Files.** `src/leads_agent/models.py`, `prompts/prompts.py`, `core/processor.py`
**Depends on.** Issue 1 · **Size.** L

---

### Issue 3 — Enforce the hard gates deterministically

**Problem.** The band and the floor are arithmetic, not judgment, but today they are paragraphs in a
system prompt. An LLM will sometimes pass a $40M company or fail a $9B one, and it has no way to
apply the floor exemption correctly because it does not know whether the account is an existing
client.

**Scope.**
- Add `src/leads_agent/core/gates.py` running **after** the scoring agent, over `ICPFit`:
  - **Band gate** — revenue outside `[min_usd, max_usd]` with confident evidence caps the score and
    sets a named disqualifier; unknown revenue does *not* disqualify, it flags "verify band first".
  - **Floor gate** — new logo below `new_logo_floor_usd` with no recurring path is disqualified;
    skipped entirely when the lead resolves to an existing client, change order or renewal
    (consumes Issue 9), or when a named strategic-investment exemption is set.
  - **Platform gate** — legacy/end-of-life with no modernization path disqualifies; every other
    tier is graduated, never a gate.
  - **Sponsor gate** — no named sponsor and no substantiated budget caps at follow-up, never
    prioritize.
  - **Caution flags** — strategy-only and standalone-governance buyers attach a warning and require
    build intent to reach prioritize; they never auto-disqualify.
- Every gate decision records `rule_id`, outcome, and the evidence it fired on, so an override is
  auditable ("exceptions are allowed; silent exceptions are not").
- Add a `manual_override` field carrying a named reason and owner.

**Acceptance criteria.**
- Table-driven tests cover each gate's fire and no-fire paths, including unknown-revenue and
  existing-client-exemption cases.
- Gate results appear in the Slack output as an explicit reason line.
- No gate can silently change a score without an emitted `rule_id`.

**Files.** `src/leads_agent/core/gates.py` (new), `core/processor.py`, `models.py`
**Depends on.** Issues 1, 2 (soft: 9) · **Size.** M

---

### Issue 4 — Eval harness and golden lead set

**Problem.** There are no tests in the repo. We just rewrote the entire ICP definition with no way to
measure whether classification got better or worse, and ICP² mandates quarterly validation against
actual wins.

**Scope.**
- Add `tests/` with pytest, and a fixture set of labeled leads: the six worked examples from the
  deck's one-sentence test slide (three qualifying, three not), plus anonymized real leads from
  collected events, plus spam/vendor-pitch negatives.
- Golden-file assertions on the structured `ICPFit` dimensions (band, door, persona, gate outcomes)
  rather than on prose.
- Offline mode: stub the search tool and the LLM so the harness runs in CI with no API keys, plus an
  opt-in live mode.
- `leads-agent eval` command reporting per-dimension accuracy, gate precision/recall, and a diff
  against the previous run; wire into `.github/workflows`.

**Acceptance criteria.**
- `pytest` passes offline with no `ANTHROPIC_API_KEY` set.
- `leads-agent eval` prints a per-dimension scoreboard and exits non-zero on regression beyond a
  configured threshold.
- Changing `prompt_config.json` and re-running shows a readable before/after diff.

**Files.** `tests/` (new), `src/leads_agent/core/eval.py` (new), `cli.py`, `pyproject.toml`,
`.github/workflows/`
**Depends on.** Issue 2 · **Size.** L

---

## Phase 1 — Deeper research

### Issue 5 — Research orchestration: real budgets, parallelism, caching, citations

**Problem.** `_research_lead()` runs one agent with one DuckDuckGo tool, and `max_searches` is only a
sentence in the prompt — nothing enforces it. Findings carry no sources, so nothing downstream can
tell an inference from a fact. Repeated leads from the same domain re-research from scratch.

**Scope.**
- Wrap the search tool in a counting/limiting decorator that hard-enforces the budget and returns a
  clear "budget exhausted" result instead of silently continuing.
- Split research into focused sub-researchers (firmographic, technographic, org/leadership, trigger
  — Issues 6–8) run concurrently, each with its own small budget, and merge into `ICPFit`.
- Add a `Source` model (url, title, retrieved_at, snippet) and require every populated `ICPFit`
  dimension to cite the sources behind it.
- Add a domain-keyed cache with a TTL for company-level findings, so contact-level research is the
  only per-lead cost on a repeat domain.
- Emit per-sub-researcher spans and token usage to Logfire.

**Acceptance criteria.**
- A lead never exceeds its configured total search budget, verified by test.
- Every non-null `ICPFit` dimension has at least one source, or is marked as inference.
- Second lead from an already-researched domain measurably cuts searches and latency.

**Files.** `src/leads_agent/agent.py`, `core/research/` (new), `models.py`, `observability.py`
**Depends on.** Issue 2 · **Size.** L

---

### Issue 6 — Firmographic research: revenue band and ownership

**Problem.** Revenue band is the primary ICP² gate, and `CompanyResearch.company_size` is a free-text
string that today gets filled with things like "mid-market". DuckDuckGo snippets are a poor source
for revenue. We also cannot detect PE backing, which is both a focus overlay and a trigger.

**Scope.**
- A firmographic sub-researcher targeting: annual revenue (with an explicit estimate range and
  confidence), employee count as a proxy, ownership type (public / private / PE-backed / portfolio
  company / non-profit / public sector), parent company, and recent corporate events.
- Add a provider interface with a DuckDuckGo-only default implementation and pluggable
  enrichment providers, so a paid data source can be dropped in without touching the agent. Document
  the interface and leave provider selection to config.
- Prefer primary sources: SEC filings and investor pages for public companies, press releases and
  PE firm portfolio pages for sponsored ones, Higher-Ed IPEDS-style disclosures and non-profit
  filings for institutions whose "revenue" is budget or endowment.
- Map the resolved figure onto the ICP² band and selling sub-band, feeding
  `ICPFit.revenue_band_fit` and the expected first-contact titles.

**Acceptance criteria.**
- On the golden set, revenue band is resolved correctly or honestly returned `unknown` — never
  guessed.
- PE-backed companies are identified with the sponsor named.
- Higher-ed and non-profit leads resolve on an appropriate budget/endowment basis rather than being
  wrongly banded.

**Files.** `src/leads_agent/core/research/firmographics.py` (new), `models.py`
**Depends on.** Issue 5 · **Size.** M

---

### Issue 7 — Technographic research: platform tier detection

**Problem.** Snowflake alignment is the go-to-market engine and the platform tier is a scored
attribute, but nothing looks for platform signals today.

**Scope.**
- A technographic sub-researcher detecting: Snowflake (in use, migrating to, or partner-directory
  listed), other modern platforms (Databricks, BigQuery, Fabric, Redshift), legacy and on-prem
  warehouses (Teradata, Netezza, Oracle, SQL Server, Hadoop), and adjacent tooling (Informatica,
  Matillion, dbt, Domo, Tableau, Power BI).
- Signal sources worth mining: engineering job postings (the highest-signal technographic source),
  partner directories and case studies, conference and summit mentions, engineering blogs, and
  earnings or annual-report language about cloud migration and data-center exit.
- Resolve to `platform_tier` 2 (Snowflake or migrating), 1 (any modern cloud platform), or 0
  (legacy, no modernization path) with cited evidence, and flag the "needs to migrate and does not
  know it yet" wedge when legacy signals coexist with an AI mandate.

**Acceptance criteria.**
- Platform tier is assigned with cited evidence, or `unknown`.
- A company with only legacy signals plus a modernization plan scores 1 or 2, not 0 — the tier is
  graduated, not a gate.
- Job-posting-derived signals are captured distinctly from marketing-page claims.

**Files.** `src/leads_agent/core/research/technographics.py` (new), `models.py`
**Depends on.** Issue 5 · **Size.** M

---

### Issue 8 — Org, leadership and trigger research

**Problem.** "Thin internal data/ML team", "named executive sponsor" and the four ICP² triggers are
scored dimensions with no research behind them. A newly appointed CDO is one of the strongest buying
signals we have and we do not look for it.

**Scope.**
- An org sub-researcher estimating data/ML org depth (team size signals, volume and seniority of
  open data-engineering and ML roles, presence or absence of a platform team) and resolving
  `internal_team_depth`. Open senior data roles are read as capacity constraint, i.e. a *positive*
  signal for us.
- Leadership mapping: current CDO / CAIO / CIO / CTO / CMO / Head of Data or DS, with appointment
  recency, and matching the inbound contact to one of the six personas including whether their title
  fits the expected band.
- Trigger detection for all four ICP² triggers: AI mandate (board/C-suite statements, earnings-call
  and annual-report language), Snowflake or platform migration, corporate events (acquisition,
  carve-out, PE transaction, newly appointed data/AI leader), and recurring operational need.
- Each trigger carries a date and a source so recency can be weighed.

**Acceptance criteria.**
- `internal_team_depth` and `conviction` are populated with cited evidence or `unknown`.
- A data/AI leader appointed within the last ~12 months is surfaced as a dated trigger.
- The inbound contact is mapped to a persona, with a mismatch called out when their title does not
  fit the company's band.

**Files.** `src/leads_agent/core/research/organization.py` (new),
`core/research/triggers.py` (new), `models.py`
**Depends on.** Issue 5 · **Size.** M

---

## Phase 2 — HubSpot integration

### Issue 9 — HubSpot read client: account context and existing-client detection

**Problem.** We treat every lead as net-new. ICP² exempts existing-client expansions, change orders
and renewals from the $150k floor and calls them our best revenue (68% win rate versus 13% for true
net-new) — so the single most valuable fact about a lead is whether we already know this account, and
we do not look it up. HubSpot already holds it.

**Scope.**
- Add `src/leads_agent/hubspot/` with an API client (private app token in `Settings`, retries, rate
  limiting, timeouts, graceful degradation when unavailable).
- Resolve an inbound lead by email and email domain to: existing contact, associated company, parent
  company, open and historical deals, lifecycle stage, and owner.
- Classify the relationship as `net_new` / `existing_client` / `former_client` /
  `open_opportunity` / `known_contact_no_deal`, and surface prior deal value, doors previously sold
  and the current owner so the lead routes to the right person.
- Feed the floor exemption in Issue 3 and add the relationship to the Slack output prominently — a
  known account should never be triaged as though it were a stranger.

**Acceptance criteria.**
- A lead from an existing client is identified as such with prior deals and owner named.
- The $150k floor gate is skipped for existing-client expansions.
- HubSpot being down or unconfigured degrades to today's behaviour with a warning, never a crash.
- Tests run against a recorded/mocked HubSpot response set; no live calls in CI.

**Files.** `src/leads_agent/hubspot/` (new), `config.py`, `core/gates.py`, `core/processor.py`
**Depends on.** Issue 2 · **Size.** L

---

### Issue 10 — Write the ICP² scorecard back to HubSpot

**Problem.** ICP² requires the scorecard to be a required field on every new opportunity: "No score,
no stage advance." Today our output lives only in a Slack thread, so it is invisible to the CRM,
un-reportable, and lost the moment the thread scrolls away.

**Scope.**
- Define the HubSpot custom-property set for ICP²: revenue band, selling sub-band, door, persona,
  overlay, platform tier, triggers, entry-shape estimate, relationship type, ICP fit score, project
  scorecard total and band, gate outcomes, missing attributes, and the one-sentence ICP statement.
- Ship a `leads-agent hubspot sync-schema` command that creates or verifies these properties
  idempotently, so the field set is version-controlled rather than hand-built in the HubSpot UI.
- Write the scorecard onto the contact (and deal, when one exists) after scoring, including a
  timestamp and the config version that produced it, so a re-scored lead is auditable.
- Guard everything behind `DRY_RUN` and a separate `HUBSPOT_WRITE_ENABLED` flag; log the intended
  payload in dry-run.

**Acceptance criteria.**
- `sync-schema` is idempotent and reports created versus existing properties.
- A scored lead has its ICP² fields populated in HubSpot, attributable to a config version.
- `DRY_RUN=true` writes nothing and logs the payload.

**Files.** `src/leads_agent/hubspot/` , `cli.py`, `core/processor.py`, `docs/DEPLOYMENT.md`
**Depends on.** Issues 2, 9 · **Size.** M

---

### Issue 11 — Ingest leads from HubSpot directly

**Problem.** `HubSpotLead.from_slack_event()` regex-parses a Slack bot message: it requires
`username == "hubspot"`, reads only `attachments[0]`, and depends on a `*Field*: Value` layout. Any
change to the HubSpot Slack workflow silently breaks ingestion, and we only ever see the handful of
fields the notification happens to include.

**Scope.**
- Add a HubSpot-native ingestion path — webhook subscription for contact creation/form submission,
  with an API-polling fallback — reading the full contact record rather than a notification summary.
- Keep the Slack path as a fallback and make the ingestion source pluggable, so this can be rolled
  out without a flag day.
- Idempotency: dedupe by HubSpot contact/submission id so a lead is never processed twice across
  both paths.
- Harden the existing parser meanwhile: scan all attachments, tolerate field-order and
  label changes, and log a clear diagnostic when a HubSpot-looking message fails to parse instead of
  returning `None` silently.

**Acceptance criteria.**
- A lead submitted in HubSpot is processed with its full field set, no Slack dependency.
- The same lead arriving via both paths is processed once.
- A malformed Slack notification logs an actionable warning naming the missing fields.

**Files.** `src/leads_agent/hubspot/`, `models.py`, `app.py`, `config.py`
**Depends on.** Issue 9 · **Size.** L

---

### Issue 12 — Account whitespace and the open-door map

**Problem.** ICP² frames existing accounts as a whitespace map — "which doors are still closed at
this client?" — and notes the floor does not apply to whitespace. Harvard is the reference case: one
logo, three buying centers. We have the door taxonomy and (after Issue 9) the deal history, so this
is mostly assembly.

**Scope.**
- Map an account's historical and open deals onto the six doors to produce an open/closed door map
  per account, including per-buying-center breakdown where deals are associated with different
  business units.
- Suggest the next door based on the ICP² wing logic: an AI-door client needs data foundations
  first; a data-door client has a natural path into the AI wing; governance rides along as a thread.
- Surface the map whenever an inbound lead resolves to an existing account, and expose
  `leads-agent whitespace <company>` for account planning outside the inbound flow.
- Flag accounts with a delivered project and no named phase two — the expansion window ICP² measures
  at a 97-day median.

**Acceptance criteria.**
- For an existing account, the reply shows which doors are open, which are closed, and a recommended
  next door with its rationale.
- `leads-agent whitespace <company>` produces the same map on demand.
- Accounts with no named phase two are listed as expansion opportunities.

**Files.** `src/leads_agent/core/whitespace.py` (new), `hubspot/`, `cli.py`, `core/processor.py`
**Depends on.** Issues 1, 9 · **Size.** M

---

## Phase 3 — GTM enablement

### Issue 13 — GTM outreach brief generator

**Problem.** The agent's output stops at a score. An AE picking up a prioritized lead still has to
work out which door this is, what to open with, what shape to propose, which accelerator applies and
which client to reference. That is the gap between "qualified" and "ready for outreach".

**Scope.**
- Generate a one-page outreach brief for every prioritized lead: the ICP² one-sentence statement,
  band and expected buying dynamic, door and wing, persona, detected triggers with dates, platform
  tier and the migration wedge if present, relationship and prior-deal context, recommended entry
  shape (archetype, size range, duration, pricing posture), applicable accelerators, matched proof
  points, and the named next phase to design in from day one.
- Drive the recommendation from the Part 2 project archetypes (Snowflake Foundation Build,
  Customer/Student/Patient 360 Phase One, Dedicated Data Team, Customer-Facing AI Experience,
  Agentic Workflow Build, Proprietary Model Build) held in config, not hardcoded.
- Include the open qualifying questions the AE still needs answered — the `missing_attributes` from
  Issue 2 turned into discovery questions.
- Deliver as a Slack thread reply (collapsible) and via `leads-agent brief <lead>`.

**Acceptance criteria.**
- Every prioritized lead gets a brief naming door, persona, entry shape and next phase.
- Archetype and accelerator recommendations trace back to config, not to code.
- The brief lists the discovery questions still outstanding.

**Files.** `src/leads_agent/core/gtm/brief.py` (new), `prompts/prompts.py`, `cli.py`,
`core/processor.py`
**Depends on.** Issues 1, 2 · **Size.** L

---

### Issue 14 — Persona messaging, objection pack and first-touch draft

**Problem.** ICP² carries, per persona, what they need, what they feel, what they object to, how to
message them and what they buy — and notes that the same $150k engagement is a CEO decision at $300M
and a VP decision at $5B. None of that reaches the person writing the first email.

**Scope.**
- Hold the persona message packs in config (Issue 1) and render, for the matched persona: the
  messaging angle, the objections to expect with responses, and what this persona actually buys.
- Draft a first-touch outreach message tuned to persona **and** band: transformation-program framing
  for a chief executive at $250M–$500M, engineering-led package framing for a functional executive at
  $500M–$1B, capacity and specialized-build framing for a VP at $1B–$10B.
- Anchor the draft in the lead's own words and the detected trigger; never invent facts about the
  company, and mark any unverified claim.
- Offer a short variant and a longer variant, and include the Snowflake co-sell angle where the
  platform tier supports it.
- Drafts are drafts: they are never sent by the agent, only surfaced for a human to send.

**Acceptance criteria.**
- Draft framing changes with band for the same persona, verified by fixture.
- Every factual claim in a draft traces to a research source or the lead's own message.
- The objection pack for the matched persona is included with the draft.

**Files.** `src/leads_agent/core/gtm/messaging.py` (new), `prompts/prompts.py`, `models.py`
**Depends on.** Issues 1, 13 · **Size.** M

---

### Issue 15 — Proof-point and accelerator matcher

**Problem.** ICP² maps each overlay to use cases, accelerators, channel and named proof points, and
the personas ask for different evidence — the Build-Capable Data Leader wants in-depth technical case
studies and explicitly "never one-pagers", while the Modernizing CIO wants migration proof points and
TCO math. We have that mapping in a deck and nowhere in the product.

**Scope.**
- Hold a proof-point library in config: client references (with permission status), the door and
  overlay each maps to, engagement shape and outcome, and the accelerator used.
- Match on overlay + door + persona, ranked by relevance, and return the *form* of evidence the
  persona wants (technical deep-dive, migration/TCO proof, board-level ROI, portfolio playbook).
- Respect reference permissions: never surface a client as a nameable reference unless the config
  marks it as such; otherwise describe it anonymously.
- Flag the gap when a lead sits in an overlay where we have no referenceable proof — that gap is
  itself a GTM signal worth reporting.

**Acceptance criteria.**
- A lead gets ranked proof points appropriate to its door, overlay and persona.
- Non-referenceable clients are never named.
- Overlays with no proof coverage are reported as gaps.

**Files.** `src/leads_agent/core/gtm/proof_points.py` (new), `prompt_config.example.json`
**Depends on.** Issues 1, 13 · **Size.** M

---

### Issue 16 — Slack actions and ICP² pipeline metrics

**Problem.** The Slack reply is read-only, so there is no way to claim a lead, ask for the brief, or
record a deliberate exception — and ICP² is explicit that "exceptions are allowed; silent exceptions
are not". Separately, the framework names the numbers it wants watched monthly and quarterly, and we
are accumulating exactly the data to produce them but reporting none of it.

**Scope.**
- Add Block Kit actions to the reply: **Claim** (assign owner, write back to HubSpot), **Brief**
  (generate Issue 13's brief on demand), **Draft outreach** (Issue 14), and **Override** — a modal
  requiring a named reason and owner, persisted to HubSpot as an auditable exception.
- Requires moving from the plain `message` handler to interactive Bolt handlers (`app.py`) with
  action acknowledgement inside Slack's 3-second window and the long work deferred.
- Add `leads-agent metrics` reporting the ICP² measures from stored scorecards: percent of pipeline
  scored as ICP, percent of new logos at $150k+, percent of SOWs with a named phase two, expansion
  rate, new-logo win rate, average deal size, and distribution across doors, overlays, bands and
  personas.
- Track override frequency by rule — a gate that is overridden constantly is a signal the doctrine or
  the config needs revisiting, which is the quarterly validation loop ICP² asks for.

**Acceptance criteria.**
- Buttons work end to end and acknowledge within Slack's timeout.
- An override cannot be recorded without a named reason and owner.
- `leads-agent metrics` reports the named ICP² measures over a date range.

**Files.** `src/leads_agent/app.py`, `core/gtm/actions.py` (new), `core/metrics.py` (new),
`cli.py`, `slack-app-manifest.yml`, `hubspot/`
**Depends on.** Issues 2, 10, 13, 14 · **Size.** L

---

## Suggested sequencing

1. **Issues 1, 2** together — the schema and the output contract; everything else depends on them.
2. **Issue 4** immediately after, so the rest of the work is measurable rather than vibes.
3. **Issue 9** early and out of order if desired: existing-client detection is the highest
   value-per-line change in the roadmap, since it corrects the floor and routes known accounts.
4. **Issue 3**, then Phase 1 (5 → 6, 7, 8 in parallel).
5. **Phase 2** (10, 11, 12), then **Phase 3** (13 → 14, 15 → 16).

## Open questions for the team

- Which paid enrichment provider, if any, for revenue and technographics (Issue 6)? The band gate is
  only as good as this input, and DuckDuckGo alone will leave a lot of `unknown`.
- Who owns the HubSpot custom-property schema, and can this app hold write scope in production
  (Issues 10, 11)?
- How should higher-ed and non-profit "revenue" be banded — operating budget, endowment, or a
  separate band definition (Issue 6)?
- Two vertical/overlay slots are marked "to be defined" and two overlay owners are unassigned in the
  deck. The config should stay easy to extend as those land.
