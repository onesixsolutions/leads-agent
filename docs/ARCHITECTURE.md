# Architecture Guide

How Leads Agent works — from lead submission to classification, research, and ICP assessment.

---

## Overview

Leads Agent is a Slack bot that:
1. Listens for HubSpot lead notifications via **Socket Mode** (WebSocket)
2. Parses contact info from the message
3. Runs a multi-stage LLM pipeline: **triage → research → ICP assessment**
4. Posts results as a threaded reply

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   HubSpot   │────▶│    Slack    │────▶│ Leads Agent  │────▶│   Claude    │
│  Workflow   │     │   Channel   │     │ (Socket Mode)│     │  (Opus 5)   │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
     Form            Bot message          Filter & parse    Triage → Research
   submission        with lead data       HubSpot messages  → ICP assess → Post
```

### The two-layer decision

The pipeline deliberately separates **evidence** from **judgement**, and separates
both from the **decision**:

| Layer | Produced by | Contains |
|-------|-------------|----------|
| Criteria | LLM (`ICPAssessment`) | Ten fixed ICP tests, each `met`/`not_met`/`partial`/`unknown` with a finding and its evidence |
| Verdict | **Code** (`icp_fit.derive_verdict`) | `in_icp` / `partial_fit` / `needs_verification` / `out_of_icp`, plus why-in / why-out / open-questions |
| Judgement | LLM (`OutreachBrief`) | The analyst's read, upside, risks, and any named exception case |

The verdict is derived in Python, never chosen by the model, so identical
evidence always produces an identical decision and every rejection names the
criterion that caused it. `unknown` is a first-class status: absent evidence
yields `needs_verification`, never a rejection.

---

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Bolt App** | `app.py` | Socket Mode connection, receives Slack events, filters HubSpot messages |
| **Processor** | `core/processor.py` | Shared pipeline: classify → format → post (used by all modes) |
| **Agent** | `agent.py` | Multi-stage LLM pipeline with pydantic-ai agents |
| **Models** | `models.py` | `HubSpotLead`, `LeadClassification`, `EnrichedLeadClassification` |
| **Prompts** | `prompts/` | Prompt configuration, ICP settings, customizable instructions |
| **ICP fit** | `icp_fit.py` | Deterministic verdict derivation from assessment criteria |
| **ICP report** | `core/icp_report.py` | Console rendering of the assessment (backtest, classify) |
| **Slack** | `slack.py` | Slack WebClient wrapper for posting messages |
| **Config** | `config.py` | Environment/`.env` settings via pydantic-settings |
| **CLI** | `cli.py` | Commands: `init`, `run`, `collect`, `backtest`, `test`, `classify`, `pull-history`, `replay` |
| **Backtest** | `core/backtest.py` | Processes collected events and runs classifier offline |
| **Classify** | `core/classify.py` | Single message classification (CLI command) |
| **Replay** | `core/replay.py` | Replay HubSpot messages from channel history |
| **History** | `core/history.py` | Fetch and save Slack channel history |
| **Init Wizard** | `core/init_wizard.py` | Interactive setup wizard for configuration |
| **Common** | `common/mask.py` | Utility for masking secrets in logs |

---

## Data Flow

### 1. HubSpot → Slack

HubSpot posts leads to Slack via workflow automation:

```
New lead from Jane Smith
Company: Acme Corp
Email: jane@acme.com
Message: We need help with AWS migration...
```

### 2. Slack → Leads Agent (Socket Mode)

The bot receives events via WebSocket (no public URL needed):

```python
# app.py handles incoming events
@app.event("message")
def handle_message(event, say, client):
    if not _is_hubspot_message(settings, event):
        return
    lead = HubSpotLead.from_slack_event(event)
    result = process_and_post(settings, lead, ...)
```

**Filtering logic:**
- Only `bot_message` subtype
- Only `username: "HubSpot"`
- Must have attachments
- Not a thread reply
- Matches `SLACK_CHANNEL_ID` (if configured)

### 3. Lead Parsing

The `HubSpotLead` model parses Slack's attachment format:

```python
class HubSpotLead(BaseModel):
    first_name: str | None
    last_name: str | None
    email: str | None
    company: str | None
    message: str | None
    raw_text: str
```

Pattern matching extracts fields from HubSpot's `*Field*: Value` format.

### 4. Multi-Stage Classification Pipeline

The `agent.py` module implements a three-stage pipeline using pydantic-ai:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLASSIFICATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Lead Input                                                         │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                    │
│  │   TRIAGE    │  Fast go/no-go decision                           │
│  │   Agent     │  Output: LeadClassification                        │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│    promising?  ─── No ──────────────────────────────┐              │
│         │                                           │              │
│        Yes                                          │              │
│         │                                           │              │
│         ▼                                           │              │
│  ┌─────────────┐                                    │              │
│  │  RESEARCH   │  Web search via DuckDuckGo        │              │
│  │   Agent     │  Finds: company info, contact role │              │
│  └──────┬──────┘                                    │              │
│         │                                           │              │
│         ▼                                           │              │
│  ┌─────────────┐                                    │              │
│  │ ICP ASSESS  │  Criteria + evidence + brief;      │              │
│  │   Agent     │  verdict derived in icp_fit.py      │              │
│  └──────┬──────┘                                    │              │
│         │                                           │              │
│         ▼                                           ▼              │
│    Post threaded reply (if not DRY_RUN)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Stage 1: Triage Agent

Fast classification to filter obvious spam/noise.

Triage decides **intent only** — never fit. A genuine inquiry from a company
we would never sell to is still `promising` here, because the ICP stage is what
judges fit and it does so with visible evidence. There is no confidence
percentage: the decision is binary.

```python
class LeadClassification(BaseModel):
    first_name: str | None
    last_name: str | None
    email: str | None
    company: str | None          # Extracted from message or email domain
    label: LeadLabel             # ignore | promising  (binary go/no-go)
    reason: str
    lead_summary: str | None     # 1-2 sentence summary
    key_signals: list[str] | None  # Tags like "budget mentioned", "vendor pitch"
```

#### Stage 2: Research Agent (Promising Leads Only)

Uses DuckDuckGo search to gather context:

```python
class CompanyResearch(BaseModel):
    company_name: str
    company_description: str
    industry: str | None
    company_size: str | None
    website: str | None
    relevance_notes: str | None

class ContactResearch(BaseModel):
    full_name: str
    title: str | None
    linkedin_summary: str | None
    relevance_notes: str | None
```

**Research strategy:**
1. Search email domain to find company website
2. Search company name for description/industry
3. Search contact name + company for role/title

#### Stage 3: ICP Assessment Agent (Promising Leads Only)

Evaluates ten fixed ICP criteria and writes the brief. The overall verdict and
action are then derived in `icp_fit.py`:

```python
class EnrichedLeadClassification(LeadClassification):
    company_research: CompanyResearch | None
    contact_research: ContactResearch | None
    research_summary: str | None

    # Model-supplied
    icp_assessment: ICPAssessment | None   # 10 criteria, each with evidence
    brief: OutreachBrief | None            # judgement layer

    # Derived in icp_fit.py — the model must not set these
    icp_verdict: ICPVerdict | None         # in_icp | partial_fit
                                           # | needs_verification | out_of_icp
    reasons_in_icp: list[str] | None
    reasons_out_of_icp: list[str] | None
    open_questions: list[str] | None
    action: LeadAction | None              # ignore | follow_up
```

**Criteria and gates** (`icp_fit.CRITERIA`):

| Criterion | Gate? | Hard? |
|-----------|-------|-------|
| Revenue band ($250M–$10B) | yes | yes |
| Deal shape ($150k+ floor) | yes | yes |
| Platform trajectory | yes | yes |
| Entry door | yes | yes |
| Executive sponsor & budget | yes | no |
| Thin internal data/ML team | yes | no |
| Trigger / urgency | no | — |
| Designed expansion path | no | — |
| Buyer persona fit | no | — |
| Focus overlay | no | — |

A gate at `not_met` disqualifies. A **hard** gate at `unknown` yields
`needs_verification`; a soft gate at `unknown` yields `partial_fit`. Non-gates
are reported but never change the verdict — the core ICP is horizontal, so an
off-overlay industry can never reject a lead.

### 5. Response

If `DRY_RUN=false`, posts a threaded reply:

```
✅ *GO* — genuine inquiry
_Clear data platform modernization inquiry_

⛔ *NOT IN ICP*
*$30M regional logistics company, no exec sponsor named, no platform signal —
no identifiable door, ~$20k ask.*

*Why not in ICP:*
• Revenue band ($250M–$10B): Self-reported $30M revenue, below the band.
• Deal shape ($150k+ floor): New logo at ~$20k with no recurring path.

*🧠 Take:* Real problem, wrong size. At $30M they cannot fund a $150k
foundation build, and the "we'll take it from there internally" framing says
they want a document, not a partner.

*Action:* `ignore`
```

---

## Run Modes

| Mode | Command | Source | Output |
|------|---------|--------|--------|
| **Production** | `run` | Socket Mode (live) | Thread replies |
| **Test** | `test` | Socket Mode (live) | Test channel |
| **Backtest** | `backtest <file>` | Collected events JSON | Console only |
| **Collect** | `collect` | Socket Mode (live) | JSON file |

### Production Mode

```bash
leads-agent run
```

Connects via Socket Mode. When HubSpot posts a lead, runs the full pipeline and posts a thread reply.

### Test Mode

```bash
leads-agent test
```

Connects via Socket Mode like production, but posts results to `SLACK_TEST_CHANNEL_ID` instead of thread replies. Good for testing the full pipeline safely.

### Collect Mode

```bash
leads-agent collect --keep 20
```

Captures raw Socket Mode events to a JSON file. Useful for inspecting event format and building test fixtures.

### Backtest Mode

```bash
leads-agent backtest collected_events.json --debug
```

Runs classifier on events from a JSON file (created by `collect`). Console-only, no Slack posts. Good for offline testing and validation.

---

## Prompt Configuration

The `prompts/` module provides customizable prompts without code changes. The main components are:
- `prompts/manager.py` - PromptManager class and configuration loading
- `prompts/prompts.py` - Prompt templates and rendering
- `prompts/utils.py` - Display utilities

### Configuration Sources

1. **`prompt_config.json`** in current directory (auto-discovered)
2. **`PROMPT_CONFIG_PATH`** environment variable
3. **Runtime updates** via `PromptManager.update_config()`

### Customizable Settings

```python
class PromptConfig(BaseModel):
    company_name: str | None              # Your company name
    services_description: str | None       # What you offer
    icp: ICPConfig | None                  # Ideal Client Profile
    qualifying_questions: list[str] | None # Custom evaluation criteria
    custom_instructions: str | None        # Additional prompt instructions
    research_focus_areas: list[str] | None # What to look for in research

class ICPConfig(BaseModel):
    description: str | None               # "Mid-market B2B SaaS"
    target_industries: list[str] | None   # ["SaaS", "FinTech"]
    target_company_sizes: list[str] | None # ["SMB", "Mid-Market"]
    target_roles: list[str] | None        # ["CTO", "VP Engineering"]
    geographic_focus: list[str] | None    # ["US", "Europe"]
    disqualifying_signals: list[str] | None # ["student", "job seeker"]
```

### Example Configuration

```json
{
  "company_name": "Acme Consulting",
  "services_description": "AI/ML consulting and custom software development",
  "icp": {
    "description": "Mid-market B2B SaaS companies",
    "target_industries": ["SaaS", "FinTech", "HealthTech"],
    "target_company_sizes": ["SMB", "Mid-Market"]
  },
  "qualifying_questions": [
    "Does this look like a real business need?",
    "Is there budget indication or enterprise context?"
  ]
}
```

### View Configuration

```bash
leads-agent prompts           # Show configuration summary
leads-agent prompts --full    # Show full rendered prompts
leads-agent prompts --json    # Output as JSON
```

---

## Slack App Configuration

### Required Scopes

| Scope | Purpose |
|-------|---------|
| `channels:history` | Read public channel messages |
| `channels:read` | View public channel info |
| `groups:history` | Read private channel messages |
| `groups:read` | View private channel info |
| `chat:write` | Post replies |
| `reactions:write` | Add ✅ / ❌ to the original lead message |

### Required Tokens

| Token | Purpose | Prefix |
|-------|---------|--------|
| `SLACK_BOT_TOKEN` | API calls (read history, post messages) | `xoxb-` |
| `SLACK_APP_TOKEN` | Socket Mode WebSocket connection | `xapp-` |

### Event Subscriptions

- `message.channels` — Public channel messages
- `message.groups` — Private channel messages

> **Important:** Bot must be invited to channels to receive events.

---

## Observability

### Logfire Integration

All agent traces are instrumented with Logfire:

```python
logfire.configure()
logfire.instrument_pydantic_ai()

with logfire.span("lead.process", lead_id=..., email=...):
    # Triage, research, scoring spans are nested here
```

**Span hierarchy:**
```
lead.process
├── lead.classify
│   ├── triage agent call
│   ├── research agent call (if promising)
│   └── scoring agent call (if promising)
└── lead.post
```

### Key Attributes

- `lead_id` — Slack thread_ts or hash of lead data
- `email` / `email_domain` — Contact info
- `company` — Extracted company name
- `label` — Triage result (intent)
- `icp_verdict` — Derived ICP determination (if promising)

---

## Deployment

### Socket Mode

The bot uses Socket Mode (outbound WebSocket), so no public URL or HTTPS setup is required. Just configure tokens and run:

```bash
docker compose up -d --build
```

### Environment Variables

```bash
# Required
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
SLACK_CHANNEL_ID=C...          # Filter to specific channel
LLM_MODEL_NAME=claude-opus-5   # Default; Opus is used for judgement quality
LLM_MAX_TOKENS=16000           # Adaptive thinking shares this budget
DRY_RUN=true                   # Don't post replies
LOGFIRE_TOKEN=...              # Observability
```

> **Model note:** the Opus 5 / Sonnet 5 generation rejects `temperature`,
> `top_p`, `top_k` and thinking `budget_tokens` with a 400. Depth is set with
> adaptive thinking plus `anthropic_effort`, per stage, in
> `agent.py::_model_settings`.

### Security Checklist

- [ ] App token has only `connections:write` scope
- [ ] Bot token not exposed in logs
- [ ] DRY_RUN tested before going live
- [ ] Logfire configured for production monitoring

---

## Module Dependency Graph

```
cli.py
  ├── app.py ────────────┐
  ├── core/backtest.py ──┼──▶ core/processor.py ──▶ agent.py ──▶ prompts/manager.py
  ├── core/classify.py ──┤         │                     │              │
  ├── core/replay.py ────┤         │                     │              ▼
  ├── core/history.py ───┤         ▼                     ▼         prompts/prompts.py
  └── core/init_wizard.py│    slack.py            models.py
                         │
  config.py ◀─────────────┘
  common/mask.py (used by config.py)
```

**Key flows:**
- `run` → `app.py` → `core/processor.py` → `agent.py`
- `test` → `app.py` (test mode) → `core/processor.py` → `agent.py`
- `backtest` → `core/backtest.py` → `agent.py` (console only)
- `classify` → `core/classify.py` → `agent.py` (direct, single message)
- `replay` → `core/replay.py` → `core/processor.py` → `agent.py`
- `pull-history` → `core/history.py` → `slack.py`
- `init` → `core/init_wizard.py` → `config.py`
