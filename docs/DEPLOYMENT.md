# Deployment Guide

Leads Agent uses **Socket Mode**, which connects to Slack via outbound WebSocket. This means:

- No public HTTPS endpoint needed
- No domain or TLS certificates required
- No inbound firewall rules to configure
- Works anywhere with outbound internet access

---

## Prerequisites

- Docker + Docker Compose
- Slack App with Socket Mode enabled ([see setup](#slack-app-setup))
- OpenAI API key (or compatible LLM endpoint)

---

## Quick Start

```bash
# Clone the repo
git clone <YOUR_REPO_URL> leads-agent
cd leads-agent

# Create .env from example
cp .env.example .env
# Edit .env with your credentials (see below)

# Start the bot
docker compose up -d --build

# View logs
docker compose logs -f primary
```

That's it. The bot connects to Slack automatically.

---

## Slack App Setup

### 1. Create the App

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App** → **From an app manifest**
3. Select your workspace
4. Paste the contents of [`slack-app-manifest.yml`](../slack-app-manifest.yml)
5. Click **Create**

### 2. Get Your Tokens

| Token | Where to Find | Env Variable |
|-------|---------------|--------------|
| Bot Token | OAuth & Permissions → Bot User OAuth Token | `SLACK_BOT_TOKEN` |
| App Token | Basic Information → App-Level Tokens → Generate | `SLACK_APP_TOKEN` |

**For the App Token:** Click "Generate Token and Scopes", name it (e.g., "socket-mode"), add scope `connections:write`, then generate.

### 3. Install to Workspace

1. Go to **Install App** in the sidebar
2. Click **Install to Workspace**
3. Authorize the permissions

### 4. Invite the Bot

In Slack, invite the bot to your leads channel:

```
/invite @Leads Agent
```

---

## Configuration

Edit `.env` with your values:

```bash
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
OPENAI_API_KEY=sk-your-openai-key

# Optional
SLACK_CHANNEL_ID=C0123456789  # Filter to specific channel
DRY_RUN=true                   # Set to false to post replies
LOGFIRE_TOKEN=                 # For observability
```

See [`.env.example`](../.env.example) for all options.

---

## Operations

### View logs

```bash
docker compose logs -f primary
```

### Update/deploy

```bash
git pull
docker compose up -d --build
```

### Restart

```bash
docker compose restart primary
```

### Stop

```bash
docker compose down
```

---

## Deployment Environments

Socket Mode works identically everywhere:

| Environment | Notes |
|-------------|-------|
| **Local machine** | Just run `docker compose up` |
| **EC2 / VPS** | No security group changes needed for Slack |
| **Behind NAT/firewall** | Works as long as outbound HTTPS is allowed |
| **Kubernetes** | Deploy as a simple pod, no ingress needed |

### EC2 Example

```bash
# SSH to your instance
ssh ec2-user@your-instance

# Clone and configure
sudo mkdir -p /opt/leads-agent
sudo chown -R "$USER" /opt/leads-agent
cd /opt/leads-agent
git clone <YOUR_REPO_URL> .

# Configure
cp .env.example .env
nano .env  # Add your tokens

# Run
docker compose up -d --build
docker compose logs -f primary
```

---

## Logfire (Observability)

Optional but recommended for production monitoring.

1. Go to [logfire.pydantic.dev](https://logfire.pydantic.dev/)
2. Create or select a project
3. **Project Settings → Write Tokens → Create Write Token**
4. Add to `.env`:

```bash
LOGFIRE_TOKEN=your-write-token
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Missing SLACK_APP_TOKEN" | Generate App-Level Token with `connections:write` scope |
| Bot not responding | Verify bot is invited to channel: `/invite @Leads Agent` |
| "Connection failed" | Check outbound HTTPS (port 443) is allowed |
| Container keeps restarting | Check logs: `docker compose logs primary` |

### Verify Slack Connection

Check logs for successful connection:

```
[STARTUP] Leads Agent (Socket Mode)
  Channel filter: C0123456789
  Dry run: true

Listening for HubSpot messages...
```

If you see errors about tokens, double-check your `.env` values.
