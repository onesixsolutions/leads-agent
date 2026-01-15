# Architecture Guide

How Leads Agent works — from lead submission to classification and enrichment.

---

## Overview

Leads Agent is a webhook service that:
1. Listens for HubSpot lead notifications in Slack
2. Parses contact info from the message
3. Classifies the lead using an LLM
4. Optionally researches promising leads via web search
5. Posts results as a threaded reply

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   HubSpot   │────▶│    Slack    │────▶│ Leads Agent  │────▶│   OpenAI    │
│  Workflow   │     │   Channel   │     │   (FastAPI)  │     │    LLM      │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
     Form           Bot message         Filter & parse       Classify
   submission       with lead data      HubSpot messages     + research
```

---

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **API** | `api.py` | Receives Slack webhooks, filters HubSpot messages, dispatches to processor |
| **Processor** | `processor.py` | Shared pipeline: classify → format → post (used by API, test, replay) |
| **Models** | `models.py` | `HubSpotLead`, `LeadClassification`, `EnrichedLeadClassification` |
| **Classifier** | `llm.py` | Classification agent + research agent with DuckDuckGo search |
| **Slack** | `slack.py` | Slack SDK wrapper, HMAC signature verification |
| **Config** | `config.py` | Environment/`.env` settings via pydantic-settings |
| **CLI** | `cli.py` | Commands: `init`, `run`, `backtest`, `test`, `replay`, `classify` |
| **Backtest** | `backtest.py` | Fetches historical HubSpot leads from Slack |

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

### 2. Slack → Leads Agent

Slack sends an event to your webhook:

```json
{
  "type": "event_callback",
  "event": {
    "type": "message",
    "subtype": "bot_message",
    "username": "HubSpot",
    "attachments": [{
      "fallback": "*First Name*: Jane\n*Last Name*: Smith\n*Email*: jane@acme.com..."
    }]
  }
}
```

**Filtering logic:**
- Only `bot_message` subtype
- Only `username: "HubSpot"`
- Must have attachments

### 3. Classification

The lead is parsed into `HubSpotLead` and sent to the LLM:

```python
class LeadClassification(BaseModel):
    first_name: str | None
    last_name: str | None
    email: str | None
    company: str | None      # Extracted from message or email domain
    label: LeadLabel         # spam | solicitation | promising
    confidence: float        # 0.0–1.0
    reason: str
```

### 4. Web Search Enrichment (Optional)

For promising leads, a research agent performs web searches:

```python
class EnrichedLeadClassification(LeadClassification):
    company_research: CompanyResearch | None
    contact_research: ContactResearch | None
    research_summary: str | None
```

**Research strategy:**
1. Search email domain to find company website
2. Broader company search for description/industry
3. Search contact name + company for role

Uses pydantic-ai's `duckduckgo_search_tool()` with configurable `max_searches` limit.

### 5. Response

If `DRY_RUN=false`, posts a threaded reply:

```
🟢 *PROMISING* (92%)
_Genuine infrastructure consulting inquiry_

📊 Company Research:
• Acme Corp: Enterprise software for supply chain management
• Industry: SaaS / Logistics
• Website: acme.com

👤 Contact Research:
• Jane Smith - VP of Engineering
```

---

## Run Modes

| Mode | Command | Source | Output | Thread? |
|------|---------|--------|--------|---------|
| **Production** | `run` | Slack webhook | Production channel | Yes |
| **Backtest** | `backtest` | Historical leads | Console only | — |
| **Test** | `test` | Historical leads | Test channel | No |
| **Replay** | `replay` | Historical leads | Production channel | Yes |

All modes share the same processing pipeline (`processor.py`).

### Production Mode

```bash
leads-agent run
```

Receives live webhooks from Slack. When HubSpot posts a lead, classifies it and posts a thread reply.

### Backtest Mode

```bash
leads-agent backtest --limit 20 --enrich --debug
```

Console-only testing. No Slack posts. Good for validating classifier behavior.

### Test Mode

```bash
leads-agent test --limit 5 --enrich
```

Posts results to `SLACK_TEST_CHANNEL_ID` (not as threads). Safe for testing Slack output format.

### Replay Mode

```bash
leads-agent replay --limit 5 --enrich --live
```

Posts results as **thread replies on original messages** in production. Use to backfill classifications on historical leads.

Features:
- Skips leads that already have replies (configurable)
- Confirmation prompt before posting
- Respects `DRY_RUN` config

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

### Event Subscriptions

- `message.channels` — Public channel messages
- `message.groups` — Private channel messages

> **Important:** Bot must be invited to channels to receive events.

---

## Classification System

### Labels

| Label | Definition |
|-------|------------|
| 🟢 **promising** | Genuine service inquiry |
| 🟡 **solicitation** | Vendors, sales, recruiters |
| 🔴 **spam** | Junk, automated, irrelevant |

### System Prompt

```
You classify inbound leads from a consulting company contact form.

Classification labels:
- spam: irrelevant, automated, SEO/link-building, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnership offers
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative — if unclear, choose spam
- Extract the company name from the message or email domain if not provided
```

### Research Prompt

```
You are researching a promising sales lead.

You have access to DuckDuckGo search tool. Use it to research:
1. The COMPANY - search for the company website/domain first
2. The CONTACT PERSON - search for their name + company

Extract:
- What does the company do?
- What industry are they in?
- What is the contact's role/title?

Be efficient - limit your searches. Do NOT make up information.
```

---

## Deployment

### Local Development

```bash
# Terminal 1
leads-agent run --reload

# Terminal 2 (expose to internet)
ngrok http 8000
# or
tailscale funnel 8000
```

### Production

```bash
# Environment
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_CHANNEL_ID=C...
OPENAI_API_KEY=sk-...
DRY_RUN=false

# Run
leads-agent run --host 0.0.0.0 --port 8000
```

**Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv pip install --system -e .
CMD ["leads-agent", "run", "--host", "0.0.0.0"]
```

### Security Checklist

- [ ] Signing secret configured and verified
- [ ] Bot token not exposed in logs
- [ ] HTTPS enforced
- [ ] DRY_RUN tested before enabling

---

## Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LEADS AGENT FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HubSpot → Slack Channel → POST /slack/events → LLM Classification      │
│                                                    │                    │
│                                                    ▼                    │
│                                         ┌─────────────────────┐         │
│                                         │  promising lead?    │         │
│                                         └─────────────────────┘         │
│                                            │              │             │
│                                           Yes            No             │
│                                            │              │             │
│                                            ▼              │             │
│                                    Web Search Research    │             │
│                                    (if --enrich)          │             │
│                                            │              │             │
│                                            ▼              ▼             │
│                                     Post threaded reply                 │
│                                     (if not DRY_RUN)                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  CLI MODES:                                                             │
│    backtest  → Console only (no Slack posts)                            │
│    test      → Post to SLACK_TEST_CHANNEL_ID (not threaded)             │
│    replay    → Post as thread replies on original messages              │
└─────────────────────────────────────────────────────────────────────────┘
```
