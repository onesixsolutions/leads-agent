<div align="center">
<!-- omit in toc -->
<picture>
  <img width="220" alt="ABC" src="docs/assets/abc.jpg">
</picture>

# HubSpot Leads Agent 🧠

<strong>AI-powered Slack bot that classifies inbound leads from HubSpot and researches promising ones.</strong>
<br>
<br>
</div align="center">

When HubSpot posts a lead to your Slack channel, Leads Agent:
1. Parses contact info (name, email, company)
2. Does a fast **go/no-go** triage (promising vs ignore)
3. If **promising**, researches the company/contact and produces a **1–5 score** + recommended action
4. Posts a threaded reply with the decision and context

---

## Quick Start

```bash
git clone https://github.com/yourusername/leads-agent.git
cd leads-agent

uv venv && source .venv/bin/activate
uv pip install -e .

leads-agent init   # Interactive setup
leads-agent run    # Start server
```

---

## Configuration

### Interactive Setup (Recommended)

```bash
leads-agent init
```

### Environment Variables

```bash
# Slack (Socket Mode)
export SLACK_BOT_TOKEN="xoxb-..."           # Bot User OAuth Token
export SLACK_APP_TOKEN="xapp-..."           # App-Level Token (connections:write)
export SLACK_CHANNEL_ID="C..."              # Production channel
export SLACK_TEST_CHANNEL_ID="C..."         # Optional: for test mode

# LLM (OpenAI by default)
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_NAME="gpt-4o-mini"         # Optional

# Behavior
export DRY_RUN="true"                       # Set to "false" to post replies
```

### Verify Configuration

```bash
leads-agent config
```

---

## Slack App Setup

### Using the Manifest (Recommended)

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From an app manifest**
3. Paste [`slack-app-manifest.yml`](slack-app-manifest.yml)
4. Install to workspace

**Get credentials:**

| Credential | Location |
|------------|----------|
| `SLACK_BOT_TOKEN` | OAuth & Permissions → Bot User OAuth Token |
| `SLACK_APP_TOKEN` | Basic Information → App-Level Tokens → Generate (scope: `connections:write`) |
| `SLACK_CHANNEL_ID` | Right-click channel → View details → Copy ID |

**Invite the bot:**

```
/invite @Leads Agent
```

> The bot uses **Socket Mode** (outbound WebSocket) — no public URL or HTTPS setup required.

---

## CLI Commands

```bash
leads-agent init                    # Setup wizard (includes prompt config)
leads-agent config                  # Show configuration
leads-agent prompts                 # Show prompt configuration
leads-agent prompts --full          # Show rendered prompts
leads-agent run                     # Start bot (Socket Mode)

# Classification
leads-agent classify "message"      # Triage; if promising, auto research + score

# Event Collection & Testing
leads-agent collect --keep 20       # Collect raw Socket Mode events
leads-agent backtest events.json    # Test classifier on collected events
leads-agent test                    # Listen via Socket Mode, post to test channel

# Debugging
leads-agent pull-history --limit 10 --print
```

### Run Modes

| Command | Description | Output |
|---------|-------------|--------|
| `run` | Production mode - reply in threads | Production (threads) |
| `test` | Test mode - post to test channel | Test channel (main) |
| `backtest` | Offline testing from collected events | Console only |

### Workflow

1. **Collect events**: `leads-agent collect --keep 20` captures raw Socket Mode events
2. **Backtest offline**: `leads-agent backtest collected_events.json` tests classifier
3. **Test live**: `leads-agent test` listens for real events, posts to test channel
4. **Go live**: `leads-agent run` production mode with thread replies

### Common Options

| Option | Description |
|--------|-------------|
| `--limit`, `-n` | Number of leads/events to process |
| `--max-searches` | Limit web searches per lead (default: 4) |
| `--dry-run` / `--live` | Override DRY_RUN config |
| `--debug`, `-d` | Show agent steps and token usage |
| `--verbose`, `-v` | Show full message history |

### Examples

```bash
# Collect events for testing
leads-agent collect --keep 10 --output hubspot_events.json

# Backtest on collected events
leads-agent backtest hubspot_events.json --debug

# Test mode - live events to test channel
leads-agent test --channel C0TEST123

# Production mode (thread replies)
leads-agent run
```

---

## LLM Configuration

### OpenAI (Default)

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_NAME="gpt-4o"  # optional, defaults to gpt-4o-mini
```

### Ollama (Local)

```bash
ollama serve
ollama pull llama3.1:8b

export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL_NAME="llama3.1:8b"
```

### Other Providers

Any OpenAI-compatible API works — set `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `OPENAI_API_KEY`.

---

## Prompt Configuration

Customize the classification behavior for your deployment without modifying code. Configure:
- **Company context** — Your company name and services
- **Ideal Client Profile (ICP)** — Target industries, company sizes, roles
- **Qualifying questions** — Custom criteria for lead evaluation
- **Research focus areas** — What to look for when enriching leads

### Configuration File

Create `prompt_config.json` in your project root (copy from [`prompt_config.example.json`](prompt_config.example.json)):

```bash
cp prompt_config.example.json prompt_config.json
# Edit with your company's ICP, questions, etc.
```

Or use `leads-agent init` to create it interactively.

The file is auto-discovered from the current directory. To use a different location:

```bash
export PROMPT_CONFIG_PATH=/path/to/my-config.json
```

### Example Configuration

```json
{
  "company_name": "Acme Consulting",
  "services_description": "AI/ML consulting and custom software development",
  "icp": {
    "description": "Mid-market B2B SaaS companies",
    "target_industries": ["SaaS", "FinTech", "HealthTech"],
    "target_company_sizes": ["SMB", "Mid-Market"],
    "target_roles": ["CTO", "VP Engineering", "Head of Data"]
  },
  "qualifying_questions": [
    "Does this look like a real business need?",
    "Is there budget indication or enterprise context?"
  ]
}
```

### View Current Configuration

```bash
leads-agent prompts           # Show configuration summary
leads-agent prompts --full    # Show full rendered prompts
leads-agent prompts --json    # Output as JSON
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config/prompts` | GET | Get current configuration and rendered prompts |
| `/config/prompts` | PUT | Replace entire configuration (runtime only) |
| `/config/prompts` | PATCH | Partially update configuration (runtime only) |
| `/config/prompts` | DELETE | Reset to file-based defaults |
| `/config/prompts/preview` | GET | Preview prompts with temporary config |

> **Note:** API updates are runtime-only and don't persist to the config file. Edit `prompt_config.json` for permanent changes.

---

## Project Structure

```
leads-agent/
├── src/leads_agent/
│   ├── api.py        # FastAPI webhook handler + config endpoints
│   ├── cli.py        # Typer CLI (init, run, collect, backtest, test, etc.)
│   ├── config.py     # Settings (pydantic-settings)
│   ├── models.py     # HubSpotLead, LeadClassification, research models
│   ├── prompts.py    # Prompt configuration and management
│   ├── llm.py        # Classification + research agents
│   ├── processor.py  # Shared processing pipeline
│   ├── backtest.py   # Historical lead fetching
│   └── slack.py      # Slack client & signature verification
├── prompt_config.example.json  # Example prompt configuration
├── docs/ARCHITECTURE.md
├── slack-app-manifest.yml
└── pyproject.toml
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Missing SLACK_APP_TOKEN" | Generate App-Level Token with `connections:write` scope |
| No classifications happening | Verify bot is invited to channel; check HubSpot is posting |
| Backtest shows no leads | Run `pull-history --print` to verify HubSpot messages exist |
| LLM errors | Check `OPENAI_API_KEY`; for Ollama ensure server is running |

---

## Tracing (Logfire)

Lead processing is wrapped in a single Logfire span (`lead.process`) so the triage/research/scoring agent traces are grouped under one lead.
In Slack-driven flows, the span uses the Slack `thread_ts` as the `lead_id` for easy correlation.

---

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** — Data flow, Slack manifest, classification system, deployment

## License

MIT — See [LICENSE](LICENSE)
