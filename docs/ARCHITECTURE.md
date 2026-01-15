# Architecture & Integration Guide

This document explains how Leads Agent works, the data flow from lead submission to classification, and how to test the system.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
  - [1. Lead Submission (HubSpot → Slack)](#1-lead-submission-hubspot--slack)
  - [2. Event Delivery (Slack → Leads Agent)](#2-event-delivery-slack--leads-agent)
  - [3. Classification (LLM)](#3-classification-llm)
  - [4. Response (Leads Agent → Slack)](#4-response-leads-agent--slack)
- [Slack App Configuration](#slack-app-configuration)
  - [Manifest Breakdown](#manifest-breakdown)
  - [OAuth Scopes Explained](#oauth-scopes-explained)
  - [Event Subscriptions](#event-subscriptions)
  - [Request Verification](#request-verification)
- [Classification System](#classification-system)
  - [Labels](#labels)
  - [LLM Prompt](#llm-prompt)
  - [Structured Output](#structured-output)
- [Backtesting](#backtesting)
  - [Purpose](#purpose)
  - [How It Works](#how-it-works)
  - [Interpreting Results](#interpreting-results)
- [Deployment Considerations](#deployment-considerations)

---

## Overview

Leads Agent is a webhook-based service that listens to a Slack channel, classifies incoming messages using an Agent, and posts classification results as threaded replies.

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   HubSpot   │────▶│    Slack    │────▶│ Leads Agent  │────▶│    Agent    │
│  (or CRM)   │     │   Channel   │     │   (FastAPI)  │     │             │
│             │     │             │◀────│              │     │             │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
     Form              Message            POST /slack/         Enrich
   submission          posted              events              message
```

---

## System Architecture

```
leads-agent/
├── src/leads_agent/
│   ├── api.py          # FastAPI app — receives Slack webhooks
│   ├── slack.py        # Slack SDK client & request verification
│   ├── llm.py          # pydantic-ai agent for classification
│   ├── domain.py       # Data models (LeadLabel, LeadClassification)
│   ├── config.py       # Environment-based settings (pydantic-settings)
│   ├── backtest.py     # Historical message testing
│   └── cli.py          # Typer CLI for setup/run/backtest
└── slack-app-manifest.yml  # Slack App configuration template
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **FastAPI (`api.py`)** | HTTP server that receives Slack event webhooks, verifies signatures, and dispatches to classifier |
| **Slack Client (`slack.py`)** | Wraps `slack_sdk` for posting messages; implements HMAC signature verification |
| **Classifier (`llm.py`)** | pydantic-ai Agent that calls an OpenAI-compatible LLM and returns structured `LeadClassification` |
| **Settings (`config.py`)** | Loads config from environment / `.env` using pydantic-settings |
| **CLI (`cli.py`)** | Typer-based CLI for `init`, `config`, `run`, `backtest`, `classify` commands |

---

## Data Flow

### 1. Lead Submission (HubSpot → Slack)

When someone submits a contact form, HubSpot (or your CRM) posts a message to Slack.

**HubSpot Workflow Setup:**

1. In HubSpot, go to **Automation → Workflows**
2. Create a workflow triggered by "Form submission"
3. Add action: **Send Slack notification**
4. Configure the message template:

```
New lead from {{contact.firstname}} {{contact.lastname}}
Company: {{contact.company}}
Email: {{contact.email}}
Message: {{contact.message}}
```

5. Select your leads channel (the one Leads Agent monitors)

**What gets posted to Slack:**

```
New lead from Jane Smith
Company: Acme Corp
Email: jane@acme.com
Message: Hi, we're looking for help migrating our infrastructure to Kubernetes. 
We have about 50 microservices and need someone with AWS/EKS experience.
```

> **Note:** You can also use Zapier, Make, or direct Slack Incoming Webhooks from any form provider.

---

### 2. Event Delivery (Slack → Leads Agent)

When a message is posted to the channel, Slack sends an HTTP POST to your server.

**Slack Event Payload:**

```json
{
  "type": "event_callback",
  "event": {
    "type": "message",
    "channel": "C0123456789",
    "user": "U9876543210",
    "text": "New lead from Jane Smith\nCompany: Acme Corp\n...",
    "ts": "1704067200.000001"
  }
}
```

**What Leads Agent does:**

```python
# api.py — simplified flow

@app.post("/slack/events")
async def slack_events(req: Request, background: BackgroundTasks):
    body = await req.body()
    
    # 1. Verify request is from Slack (HMAC signature)
    if not verify_slack_request(settings, req, body):
        return {"error": "Invalid request"}
    
    payload = await req.json()
    
    # 2. Handle Slack URL verification (one-time setup)
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}
    
    event = payload.get("event", {})
    
    # 3. Filter: only process new messages in our channel
    if not _is_relevant_slack_message_event(settings, event):
        return {"ok": True}
    
    # 4. ACK immediately, process in background (Slack 3s timeout)
    background.add_task(_handle_message_event, settings, event)
    
    return {"ok": True}
```

**Why background processing?**

Slack expects a response within 3 seconds. LLM inference can take longer, so we:
1. Immediately return `{"ok": True}` to Slack
2. Process the classification in a background task
3. Post the reply asynchronously

---

### 3. Classification (LLM)

The message text is sent to an LLM with a structured output schema.

**LLM Request:**

```python
# llm.py

SYSTEM_PROMPT = """
You classify inbound leads from a consulting company contact form.

Definitions:
- spam: irrelevant, automated, SEO, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnerships
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative
- If unclear, choose spam
- Provide a short reason
"""

# Using pydantic-ai for structured output
classifier = Agent(
    model=model,
    result_type=LeadClassification,  # Enforces JSON schema
    system_prompt=SYSTEM_PROMPT,
)

result = classifier.run_sync(f'Message:\n"""\n{text}\n"""')
```

**LLM Response (structured):**

```json
{
  "label": "promising",
  "confidence": 0.92,
  "reason": "Genuine infrastructure consulting inquiry with specific technical requirements"
}
```

**Structured Output with pydantic-ai:**

The `LeadClassification` model ensures the LLM returns valid JSON:

```python
# domain.py

class LeadLabel(str, Enum):
    spam = "spam"
    solicitation = "solicitation"
    promising = "promising"

class LeadClassification(BaseModel):
    label: LeadLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
```

---

### 4. Response (Leads Agent → Slack)

If `DRY_RUN=false`, the bot posts a threaded reply:

```python
# api.py

client.chat_postMessage(
    channel=event["channel"],
    thread_ts=event["ts"],  # Reply in thread, not main channel
    text=f"🧠 Lead classification: *{result.label}* ({result.confidence:.2f})\n_{result.reason}_",
)
```

**What appears in Slack:**

```
┌──────────────────────────────────────────────────────────┐
│ New lead from Jane Smith                                 │
│ Company: Acme Corp                                       │
│ Email: jane@acme.com                                     │
│ Message: Hi, we're looking for help migrating our...     │
│                                                          │
│ ┌─ Thread ────────────────────────────────────────────┐  │
│ │ 🧠 Lead classification: *promising* (0.92)          │  │
│ │ _Genuine infrastructure consulting inquiry with     │  │
│ │ specific technical requirements_                    │  │
│ └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Slack App Configuration

### Manifest Breakdown

The `slack-app-manifest.yml` declaratively configures your Slack App:

```yaml
display_information:
  name: Leads Classifier           # App name shown in Slack
  description: AI-powered lead classification bot
  background_color: "#1a1a2e"      # App icon background

features:
  bot_user:
    display_name: Leads Classifier # Bot username (@Leads Classifier)
    always_online: true            # Show green dot (cosmetic only)

oauth_config:
  scopes:
    bot:                           # Permissions the bot requests
      - channels:history           # Read messages in public channels (bot must be invited)
      - channels:read              # See public channel metadata
      - groups:history             # Read messages in private channels (bot must be invited)
      - groups:read                # See private channel metadata
      - chat:write                 # Post messages

settings:
  event_subscriptions:
    request_url: https://YOUR_DOMAIN/slack/events  # Your webhook URL
    bot_events:
      - message.channels           # Messages in public channels (bot must be member)
      - message.groups             # Messages in private channels (bot must be member)
```

> **Important:** The bot only receives messages from channels it's been invited to. This applies to both public and private channels. After installing the app, you must invite the bot to each channel where you want it to operate.

### OAuth Scopes Explained

| Scope | Why Needed |
|-------|------------|
| `channels:history` | Read messages in public channels the bot is invited to (required for backtesting + receiving events) |
| `channels:read` | Access basic public channel info (name, ID) — used by some SDK methods |
| `groups:history` | Read messages in private channels the bot is invited to |
| `groups:read` | Access basic private channel info (name, ID) |
| `chat:write` | Post messages as the bot (classification replies) |

> **Note:** The `groups:*` scopes are Slack's terminology for private channels. Despite the naming, this does not grant access to DMs or group DMs—only private channels where the bot is explicitly invited.

### Event Subscriptions

Slack uses a **push model** — when something happens, Slack POSTs to your URL.

```
┌─────────────┐         ┌──────────────────┐
│    Slack    │  POST   │   Your Server    │
│   Platform  │────────▶│ /slack/events    │
│             │         │                  │
│  (detects   │         │  (receives JSON  │
│   message)  │         │   event payload) │
└─────────────┘         └──────────────────┘
```

**URL Verification:**

When you first set the Request URL, Slack sends a challenge:

```json
{
  "type": "url_verification",
  "challenge": "abc123xyz"
}
```

Your server must respond:

```json
{
  "challenge": "abc123xyz"
}
```

This proves you control the endpoint.

### Request Verification

Every request from Slack is signed with your **Signing Secret**.

**Verification flow:**

```python
# slack.py

def verify_slack_request(settings, req, body):
    timestamp = req.headers.get("X-Slack-Request-Timestamp")
    signature = req.headers.get("X-Slack-Signature")
    
    # 1. Reject old requests (replay attack prevention)
    if abs(time.time() - int(timestamp)) > 300:  # 5 minutes
        return False
    
    # 2. Compute expected signature
    basestring = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(
        signing_secret.encode(),
        basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # 3. Constant-time comparison (timing attack prevention)
    return hmac.compare_digest(expected, signature)
```

**Why this matters:**

Without verification, anyone could POST fake events to your endpoint and trigger classifications or extract information.

---

## Classification System

### Labels

| Label | Definition | Examples |
|-------|------------|----------|
| 🟢 **promising** | Genuine service inquiry or collaboration | "We need help with AWS migration", "Interested in consulting rates" |
| 🟡 **solicitation** | Vendors, sales, recruiters, partnerships | "We offer SEO services", "Partnership opportunity", "Open to new roles?" |
| 🔴 **spam** | Junk, automated, irrelevant | "Buy crypto now", "You've won!", random gibberish |

### LLM Prompt

The system prompt is intentionally brief and rule-based:

```
You classify inbound leads from a consulting company contact form.

Definitions:
- spam: irrelevant, automated, SEO, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnerships
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative
- If unclear, choose spam
- Provide a short reason
```

**Design decisions:**

1. **Conservative default:** Ambiguous messages → spam. Better to manually review than miss a solicitation.
2. **Short reason:** Forces the model to be concise; reasons appear in Slack.
3. **No few-shot examples:** Keeps token count low; structured output handles formatting.

### Structured Output

Using `pydantic-ai` ensures the LLM returns valid, typed data:

```python
class LeadClassification(BaseModel):
    label: LeadLabel              # Enum: spam | solicitation | promising
    confidence: float             # 0.0–1.0, validated by Pydantic
    reason: str                   # Short explanation
```

The agent automatically:
- Injects the JSON schema into the prompt
- Parses the response
- Validates against the schema
- Retries if the LLM returns invalid JSON

---

## Backtesting

### Purpose

Before enabling live responses (`DRY_RUN=false`), you should verify the classifier works well on your actual messages.

Backtesting lets you:
1. **Evaluate accuracy** on historical data
2. **Tune the prompt** based on misclassifications
3. **Spot edge cases** (e.g., legitimate messages marked as spam)

### How It Works

```python
# backtest.py — simplified

def run_backtest(settings, limit=50):
    # 1. Fetch recent messages from Slack
    client = slack_client(settings)
    resp = client.conversations_history(
        channel=settings.slack_channel_id,
        limit=limit
    )
    
    # 2. Filter to top-level messages only
    for msg in resp.get("messages", []):
        if msg.get("subtype"):      # Skip system messages
            continue
        if msg.get("thread_ts"):    # Skip thread replies
            continue
        
        text = msg.get("text", "")
        
        # 3. Classify each message
        result = classify_message(settings, text)
        
        # 4. Print results
        print("-" * 60)
        print(text)
        print(f"→ {result.label} ({result.confidence:.2f})")
        print(f"Reason: {result.reason}")
```

**Running a backtest:**

```bash
leads-agent backtest --limit 20
```

**Sample output:**

```
Backtesting last 20 messages

------------------------------------------------------------
New lead from John Doe
Company: Tech Startup
Email: john@startup.io
Message: Looking for DevOps consulting, specifically around CI/CD pipelines
→ promising (0.94)
Reason: Genuine DevOps consulting inquiry with specific requirements

------------------------------------------------------------
Hi! We're an SEO agency and can help you rank #1 on Google.
Contact us at seo@spammy.com for a free audit!
→ solicitation (0.89)
Reason: SEO vendor sales pitch

------------------------------------------------------------
🚀 CRYPTO OPPORTUNITY! 10X YOUR INVESTMENT NOW 🚀
→ spam (0.98)
Reason: Cryptocurrency promotion spam
```

### Interpreting Results

| Scenario | Action |
|----------|--------|
| Promising lead marked as spam | Review prompt; may need to loosen criteria |
| Solicitation marked as promising | Review prompt; add vendor patterns to definitions |
| Low confidence (<0.7) | Message is ambiguous; consider manual review workflow |
| Consistent misclassifications | Add few-shot examples or fine-tune the model |

**Iterating on the prompt:**

1. Run backtest
2. Identify misclassifications
3. Adjust `SYSTEM_PROMPT` in `llm.py`
4. Re-run backtest
5. Repeat until satisfied

---

## Deployment Considerations

### Local Development

```bash
# Terminal 1: Start the server
leads-agent run --reload

# Terminal 2: Expose via ngrok
ngrok http 8000
# Copy the https://xxx.ngrok.io URL to Slack manifest
```

### Production

**Requirements:**
- HTTPS endpoint (Slack requires TLS)
- Public IP or domain
- Process manager (systemd, Docker, etc.)

**Example with Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv pip install --system -e .
CMD ["leads-agent", "run", "--host", "0.0.0.0", "--port", "8000"]
```

**Environment variables in production:**

```bash
# Required — Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_CHANNEL_ID=C...

# Required — LLM (OpenAI by default)
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini

# Optional — for Ollama or other OpenAI-compatible providers
# LLM_BASE_URL=http://localhost:11434/v1

# Enable responses
DRY_RUN=false
```

### Security Checklist

- [ ] Signing secret is set and requests are verified
- [ ] Bot token is not exposed in logs
- [ ] HTTPS is enforced
- [ ] Rate limiting is in place (consider adding to FastAPI)
- [ ] DRY_RUN tested before enabling live responses

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LEADS AGENT FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ HubSpot  │───▶│  Slack   │───▶│ Leads Agent  │───▶│ LLM (OpenAI/     │  │
│  │ Workflow │    │ Channel  │    │ POST /slack/ │    │ Ollama)          │  │
│  │          │    │          │◀───│ events       │◀───│                  │  │
│  └──────────┘    └──────────┘    └──────────────┘    └──────────────────┘  │
│       │               │                │                      │            │
│   Form submit    Message posted    Verify sig +          Classify          │
│                                    ACK quickly            message          │
│                                         │                      │            │
│                                    Background task        Return JSON      │
│                                         │                      │            │
│                                    Post threaded      {label, confidence,  │
│                                    reply (if not       reason}             │
│                                    DRY_RUN)                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  BACKTEST MODE: Fetch historical messages → Classify → Print results       │
│  (No Slack posts, just console output for evaluation)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

For setup instructions, see the main [README.md](../README.md).
