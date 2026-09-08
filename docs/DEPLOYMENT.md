# Deployment Guide

Leads Agent uses **Socket Mode**, which connects to Slack via outbound WebSocket. This means:

- No public HTTPS endpoint needed
- No domain or TLS certificates required
- No inbound firewall rules to configure
- Works anywhere with outbound internet access

The one exception is the optional [lead briefs](#lead-briefs-html-pages-in-s3)
feature, which serves HTML pages over a port you choose to publish. It is off
by default; with `BRIEFS_ENABLED` unset, everything above still holds.

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

## Lead Briefs (HTML pages in S3)

> **Provisioned and verified** — `us-west-2`, bucket **`onesix-leads-agent`**.
> Public access fully blocked, SSE-S3 default encryption, versioning enabled as
> a backstop, tagged `contains-pii=true`.
>
> The EC2 instance role **`strong-automation`** was tested against it and
> already grants everything the app needs — `PutObject`, `GetObject`,
> `HeadObject`, `ListBucket`. Critically, a request for a **missing** key
> returns `404` rather than `403`, which is what lets the version allocator
> distinguish a free slot from a permission error. **No IAM change is required.**
>
> Briefs name and characterise real people, so publish the port on the tailnet
> via `BRIEFS_BIND_ADDR`, never on a public interface.


The Slack card carries the decision; the **brief** carries the evidence — the
full ICP assessment, the judgement layer and the research, as a styled HTML
page. Every analysis is stored as a new version, so you can always go back and
see what an earlier run concluded.

This is entirely optional. With `BRIEFS_ENABLED` unset the bot behaves exactly
as it did before: no S3 calls, no listener, no new failure modes.

### What gets stored

```
s3://<bucket>/<prefix>/<lead_id>/index.json     pointer + version log
s3://<bucket>/<prefix>/<lead_id>/v0001.html     the brief, as served
s3://<bucket>/<prefix>/<lead_id>/v0001.json     the classification, as data
s3://<bucket>/<prefix>/<lead_id>/v0002.html     the next analysis
...
```

- **Versions are object paths, not S3 bucket versions.** You can list them,
  link to them and diff them without touching version ids. Bucket versioning
  is not required and does not need to be enabled.
- **Version numbers are zero-padded** so `aws s3 ls` returns them in order.
- **Nothing is ever overwritten or deleted.** A re-analysed lead gets a new
  version; `index.json` moves the "current" pointer.
- **The JSON alongside each page is the point of the exercise long-term.** The
  prose answers "what about this lead?"; the JSON answers "which ICP gate
  fails most often?" without anyone re-parsing a rendered page.

`<lead_id>` is derived from the lead's own identity (email, else company +
name) as `<slug>-<10 hex digits>`. It is stable across re-analysis — that is
what makes the history accumulate — and it deliberately does not contain the
contact's email address, because these links get pasted into Slack.

### Link mode

`BRIEFS_LINK_MODE` decides what the Slack card links to:

| Mode | Opens from | Expires |
|------|-----------|---------|
| `presigned` (default) | anywhere, nothing needs to be running | **yes** — with the instance-role session token, typically hours |
| `app` | only where the listener is reachable (the tailnet) | no |

Presigned is the default because it works before the listener is deployed. If
you need links that survive in Slack scrollback, switch to `app` once the
listener is published on the tailnet, or presign with static IAM user
credentials instead of the instance role.

### URLs

| URL | Serves |
|-----|--------|
| `/briefs/<lead_id>` | The current brief. **This is the link that goes into Slack** — it never goes stale. |
| `/briefs/<lead_id>/v3` | A specific version. |
| `/briefs/<lead_id>/history` | Every version of this lead, newest first, with verdict and timestamp. |
| `/briefs/<lead_id>.json` | The current structured record. |
| `/briefs/<lead_id>/v3.json` | A specific version's structured record. |
| `/healthz` | Liveness probe. |

To get back to a previous brief: open the current one and follow **All
versions** in the footer, or go straight to `/briefs/<lead_id>/history`.

### The HTTP server

Socket Mode is outbound-only and the app had no web server, so briefs are
served by a `http.server.ThreadingHTTPServer` running in a **daemon thread**
alongside the Socket Mode loop. `SocketModeHandler.start()` blocks the main
thread forever, so the listener is started first; being a daemon thread means
Ctrl+C still stops the process and the listener can never be the reason the
bot fails to exit.

It is the standard library rather than FastAPI/Flask on purpose: the whole
surface is four read-only GET routes serving bytes that already exist in S3,
and nothing in the dependency tree can listen on a socket (`slack-bolt` brings
no server in Socket Mode, `httpx` is a client). A framework would mean a new
dependency and a second runtime model for four routes.

A bind failure is logged and swallowed. Briefs are a convenience; classifying
leads is the job.

### Port and network exposure

**Port 8080.** On the current host, `80` is taken by `strongbot` and `5678` by
`n8n`; 8080 is free.

`docker-compose.yml` publishes the port as:

```yaml
ports:
  - "${BRIEFS_BIND_ADDR:-127.0.0.1}:${BRIEFS_HTTP_PORT:-8080}:${BRIEFS_HTTP_PORT:-8080}"
```

**The default is loopback, so an unconfigured host exposes nothing.**
`BRIEFS_HTTP_HOST` (`0.0.0.0`) is the bind address *inside* the container and
should be left alone — who can reach it is decided by `BRIEFS_BIND_ADDR` on
the host side.

**Serve over Tailscale, not the public internet.** A brief names an individual,
names their employer, and states a pointed judgement about both. The host is
already on the tailnet at `100.79.160.6`, so:

```bash
# in .env on the server
BRIEFS_BIND_ADDR=100.79.160.6
BRIEFS_BASE_URL=http://100.79.160.6:8080
```

Anyone on the tailnet can then click the link in Slack; nobody else can reach
it, and no security group change, TLS certificate, nginx or Caddy is needed.
The server is read-only and unauthenticated by design: exposure is a network
decision, not something the app tries to solve with a token. If briefs ever
need to be readable from outside the tailnet, put a reverse proxy with real
auth in front of it — do not simply widen `BRIEFS_BIND_ADDR` to `0.0.0.0`.

Consider a Tailscale MagicDNS name in `BRIEFS_BASE_URL` (e.g.
`http://strong-automation:8080`) rather than the raw IP, so links survive the
host changing address.

### AWS credentials

Credentials are **never** read from `.env` or from `Settings`. The S3 client
uses the default boto3 chain, in order: EC2 instance role → environment
variables → `~/.aws` profile. On the EC2 host the instance role is the
intended path — attach the policy below to it and set no AWS variables at all.

If the instance has no role, the fallback is standard AWS environment
variables in `.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`); `env_file` passes them through to boto3 untouched.

### Minimum IAM policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadWriteBriefObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::YOUR-BUCKET/briefs/*"
    },
    {
      "Sid": "DistinguishMissingFromForbidden",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::YOUR-BUCKET"
    }
  ]
}
```

Two notes on the second statement, which is not optional:

- Without `s3:ListBucket`, S3 answers a request for a **missing** object with
  `403 AccessDenied` instead of `404 NoSuchKey`. The version allocator relies
  on being able to tell "this version does not exist yet" from "I am not
  allowed to look", so without it every publish fails (safely — it returns no
  link rather than overwriting anything, but no brief is written).
- It is granted on the bucket with **no `s3:prefix` condition**, because
  `HeadObject`/`GetObject` do not supply `s3:prefix` and a conditioned grant
  would not apply to them.

`s3:DeleteObject` is deliberately absent. Briefs are append-only; that is what
makes the history trustworthy.

Bucket settings: Block Public Access **on** (the app reads the objects with
its own credentials and serves them itself — the bucket is never public),
default SSE-S3 encryption, no bucket versioning needed. A lifecycle rule
expiring `briefs/` after N years is reasonable; do not add one that expires
noncurrent versions, as there are none.

### Enabling it

```bash
# .env on the server
BRIEFS_ENABLED=true
BRIEFS_S3_BUCKET=onesix-leads-agent
BRIEFS_S3_REGION=us-west-2
BRIEFS_BASE_URL=http://100.79.160.6:8080
BRIEFS_BIND_ADDR=100.79.160.6
```

```bash
docker compose up -d --build   # rebuild: boto3 is a new dependency
curl -s http://127.0.0.1:8080/healthz     # -> ok
leads-agent config                         # confirms what was picked up
```

### Brief troubleshooting

| Symptom | Cause |
|---------|-------|
| No link in Slack, `BRIEFS_ENABLED is set but BRIEFS_S3_BUCKET is empty` | Set the bucket. |
| No link, `Failed to publish lead brief` in logs | Check the traceback — usually IAM (see the `s3:ListBucket` note above) or the wrong region. |
| Link is `/briefs/...` with no host | `BRIEFS_BASE_URL` is unset and the bind host is a wildcard. Set `BRIEFS_BASE_URL`. |
| `Could not bind brief HTTP server` | Port in use inside the container; change `BRIEFS_HTTP_PORT`. |
| Link times out from a laptop | Not on the tailnet, or `BRIEFS_BIND_ADDR` is still `127.0.0.1`. |
| `docker compose up` fails on port allocation | Something else took `BRIEFS_HTTP_PORT` on the host. |

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
[STARTUP] Leads Agent
  Channel filter: C0123456789
  Dry run: true

Listening for HubSpot messages... (Ctrl+C to stop)
```

If you see errors about tokens, double-check your `.env` values.
