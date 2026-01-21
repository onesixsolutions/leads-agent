# Deployment (EC2 + Docker + optional Tailscale Funnel)

This service is a small API that Slack calls at:

- `POST /slack/events`

## Prereqs

- An existing **EC2 instance** with a **static IP**
- Docker installed (Docker Engine + Compose plugin)
- A Slack App configured (see `README.md`)
- Optional: Tailscale (only if using Funnel / Tailscale SSH)

## One-time setup (on the EC2 host)

```bash
sudo mkdir -p /opt/leads-agent
sudo chown -R "$USER":"$USER" /opt/leads-agent
cd /opt/leads-agent

git clone <YOUR_REPO_GIT_URL> .

# Create /opt/leads-agent/.env with required secrets (see README for env vars)
chmod 600 .env

docker compose up -d --build
curl -f http://127.0.0.1:8000/
```

Notes:

- By default `docker-compose.yml` binds the API to `127.0.0.1:8000` (private to the host).
- The container auto-restarts (`restart: unless-stopped`).

## Expose HTTPS for Slack (pick one)

Slack requires a **public HTTPS** URL. Set Slack’s Events Request URL to:

- `https://YOUR_HOSTNAME/slack/events`

### Option A (recommended): static IP + domain + HTTPS (Caddy)

1. Create a DNS **A record**:
   - `your-domain.example` → your EC2 static IP
2. Allow inbound **80/443** to the instance (security group managed elsewhere)
3. Enable Caddy:
   - In `docker-compose.yml`, uncomment the `caddy` service
   - In `deploy/Caddyfile`, replace `your-domain.example` with your real hostname
4. Apply:

```bash
cd /opt/leads-agent
docker compose up -d --build
```

### Option B: Tailscale Funnel (if you can’t open inbound 80/443)

1. Install Tailscale on the EC2 host and join the tailnet:

```bash
tailscale up ...
```

2. Use Tailscale **Serve** + **Funnel** to publish HTTPS → `http://127.0.0.1:8000`.

Because the CLI differs by version, use the host’s help to get the exact commands:

```bash
tailscale version
tailscale serve --help
tailscale funnel --help
```

This repo previously used the pattern:

- `tailscale funnel 8000`

Once enabled, set Slack’s Request URL to:

- `https://YOUR_TAILSCALE_PUBLIC_HOSTNAME/slack/events`

## Deploy/update

```bash
cd /opt/leads-agent
git pull
docker compose up -d --build
docker compose logs -f leads-agent
```

## Rollback

```bash
cd /opt/leads-agent
git checkout <previous_sha>
docker compose up -d --build
```

## Production note (enrichment)

In webhook/API mode (`leads-agent run`), enrichment is **disabled by default**. If you want enrichment in production, change `src/leads_agent/api.py` and plan for extra latency/cost.

