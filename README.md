# 🧠 Leads Agent

AI-powered Slack bot that automatically classifies inbound leads from your contact form.

**Labels:**
- 🟢 **promising** — Genuine inquiry about services or collaboration
- 🟡 **solicitation** — Vendors, sales pitches, recruiters, partnerships
- 🔴 **spam** — Irrelevant, automated, SEO, crypto, junk

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/yourusername/leads-agent.git
cd leads-agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# Run the setup wizard
leads-agent init

# Start the server
leads-agent run
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager
- **[OpenAI API key](https://platform.openai.xcom/api-keys)** — For LLM inference (or use Ollama locally)

### Install with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .

# Verify installation
leads-agent --help
```

---

## Configuration

### Option A: Interactive Setup (Recommended)

```bash
leads-agent init
```

This wizard will prompt you for all required values and create a `.env` file.

### Option B: Manual Setup

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Option C: Environment Variables

Export directly in your shell or CI/CD:

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_SIGNING_SECRET="..."
export SLACK_CHANNEL_ID="C..."
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_NAME="gpt-4o-mini"
export DRY_RUN="true"
```

### Verify Configuration

```bash
leads-agent config
```

---

## Slack App Setup

### Option A: Using the Manifest (Recommended)

The fastest way to set up the Slack App:

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From an app manifest**
3. Select your workspace
4. Paste the contents of [`slack-app-manifest.yml`](slack-app-manifest.yml)
5. **Replace `YOUR_DOMAIN`** with your actual server URL (e.g., `https://example.com` or your ngrok URL)
6. Click **Create**
7. Go to **Install App** and click **Install to Workspace**

Then grab your credentials:

| Credential | Location |
|------------|----------|
| `SLACK_BOT_TOKEN` | **OAuth & Permissions** → Bot User OAuth Token (starts with `xoxb-`) |
| `SLACK_SIGNING_SECRET` | **Basic Information** → App Credentials → Signing Secret |
| `SLACK_CHANNEL_ID` | In Slack: right-click the channel → **View channel details** → copy the ID |

Skip to [Invite the Bot](#step-6-invite-the-bot).

---

### Option B: Manual Setup

#### Step 1: Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From scratch**
3. Name it (e.g., "Leads Classifier") and select your workspace

#### Step 2: Configure Bot Permissions

1. Navigate to **OAuth & Permissions** in the sidebar
2. Under **Scopes → Bot Token Scopes**, add:
   - `channels:history` — Read messages in public channels the bot is invited to
   - `channels:read` — View basic public channel info
   - `groups:history` — Read messages in private channels the bot is invited to
   - `groups:read` — View basic private channel info
   - `chat:write` — Send messages as the bot
3. Click **Install to Workspace** and authorize

#### Step 3: Get Your Credentials

| Credential | Location |
|------------|----------|
| `SLACK_BOT_TOKEN` | **OAuth & Permissions** → Bot User OAuth Token (starts with `xoxb-`) |
| `SLACK_SIGNING_SECRET` | **Basic Information** → App Credentials → Signing Secret |
| `SLACK_CHANNEL_ID` | In Slack: right-click the channel → **View channel details** → copy the ID at the bottom |

#### Step 4: Enable Event Subscriptions

1. Navigate to **Event Subscriptions** in the sidebar
2. Toggle **Enable Events** to ON
3. Set **Request URL** to your server's public endpoint:
   ```
   https://your-domain.com/slack/events
   ```
   > 💡 For local development use
   >  - [ngrok](https://ngrok.com/): `ngrok http 8000`
   >  - [tailscale](https://tailscale.com/): `tailscale funnel 8000`
4. Slack will send a verification challenge — your server must be running!

#### Step 5: Subscribe to Bot Events

Under **Subscribe to bot events**, click **Add Bot User Event** and add:

- `message.channels` — Messages in public channels (bot must be invited)
- `message.groups` — Messages in private channels (bot must be invited)

Click **Save Changes**.

#### Step 6: Invite the Bot

**Important:** The bot only receives messages from channels it's been invited to. This works for both public and private channels.

In Slack, invite the bot to your leads channel:

```
/invite @Leads Classifier
```

> **Note:** For private channels, you must have the bot invited before it can receive any messages. The bot will not have access to message history from before it was invited.

---

## Usage

### CLI Commands

```bash
# Interactive setup wizard
leads-agent init

# Show current configuration
leads-agent config

# Start the API server
leads-agent run [--host 0.0.0.0] [--port 8000] [--reload]

# Backtest against historical messages
leads-agent backtest [--limit 50]

# Classify a single message (for testing)
leads-agent classify "Hi, I'm interested in your consulting services..."
```

### Running the Server

```bash
# Development (with auto-reload)
leads-agent run --reload

# Production
leads-agent run --host 0.0.0.0 --port 8000
```

The server exposes a single endpoint:

- `POST /slack/events` — Receives Slack event payloads

### Dry Run Mode

By default, `DRY_RUN=true` — the bot logs classifications but doesn't post replies. 

To enable posting:

```bash
export DRY_RUN=false
leads-agent run
```

---

## LLM Configuration

### OpenAI (Default)

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Optionally change the model (default: gpt-4o-mini)
export LLM_MODEL_NAME="gpt-4o"
```

### Local (Ollama)

```bash
# Install and run Ollama
ollama serve

# Pull a model
ollama pull llama3.1:8b

# Configure for Ollama
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL_NAME="llama3.1:8b"
# No OPENAI_API_KEY needed for Ollama
```

### Other Providers

Any OpenAI-compatible API works. Set `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `OPENAI_API_KEY` accordingly.

---

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run with auto-reload
leads-agent run --reload

# Run backtest to verify classifier behavior
leads-agent backtest --limit 20
```

### Project Structure

```
leads-agent/
├── src/leads_agent/
│   ├── __init__.py      # Package exports
│   ├── __main__.py      # python -m leads_agent
│   ├── api.py           # FastAPI application
│   ├── backtest.py      # Historical message testing
│   ├── cli.py           # Typer CLI
│   ├── config.py        # Settings via pydantic-settings
│   ├── domain.py        # Data models
│   ├── llm.py           # LLM agent for classification
│   └── slack.py         # Slack client helpers
├── docs/
│   └── ARCHITECTURE.md  # Detailed architecture & integration guide
├── main.py              # Convenience shim
├── pyproject.toml       # Package configuration
├── slack-app-manifest.yml  # Slack App manifest template
├── .env.example         # Example environment file
└── README.md
```

### Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** — Deep dive into the system design, data flow (HubSpot → Slack → LLM), Slack manifest explained, and backtesting

---

## Troubleshooting

### "Invalid request" on Slack events

- Verify `SLACK_SIGNING_SECRET` is correct
- Ensure your server clock is synchronized (Slack rejects requests >5 min old)

### "channel_not_found" errors

- Verify `SLACK_CHANNEL_ID` is correct (should start with `C`)
- Ensure the bot is invited to the channel

### No classifications happening

- Check that Event Subscriptions are enabled in your Slack App
- Verify the bot is subscribed to `message.channels` and/or `message.groups`
- **Ensure the bot is invited to the channel** — the bot only receives events from channels it's a member of
- For private channels: confirm `groups:history` and `groups:read` scopes are added
- Check server logs for incoming events

### LLM connection errors

- Verify `OPENAI_API_KEY` is set and valid
- For Ollama: ensure `ollama serve` is running and `LLM_BASE_URL` is set
- Check that `LLM_MODEL_NAME` is a valid model for your provider

---

## License

MIT — See [LICENSE](LICENSE) for details.
