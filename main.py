import hashlib
import hmac
import os
import time
from enum import Enum
from typing import Iterable

from fastapi import Request
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from slack_sdk import WebClient

### Configuration

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama3.1:8b")

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

### Data Model


class LeadLabel(str, Enum):
    spam = "spam"
    solicitation = "solicitation"
    promising = "promising"


class LeadClassification(BaseModel):
    label: LeadLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


### LLM Agent

model = OpenAIModel(
    base_url=LLM_BASE_URL,
    api_key="local",
    model_name=LLM_MODEL_NAME,
)

classifier = Agent(
    model=model,
    result_type=LeadClassification,
    system_prompt="""
You classify inbound leads from a consulting company contact form.

Definitions:
- spam: irrelevant, automated, SEO, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnerships
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative
- If unclear, choose spam
- Provide a short reason
""",
)


def classify_message(text: str) -> LeadClassification:
    result = classifier.run_sync(
        f"""
Message:
\"\"\"
{text}
\"\"\"
"""
    )
    return result.data


### Slack helpers
slack_client = WebClient(token=SLACK_BOT_TOKEN)


def verify_slack_request(req: Request, body: bytes) -> bool:
    timestamp = req.headers.get("X-Slack-Request-Timestamp")
    signature = req.headers.get("X-Slack-Signature")

    if not timestamp or not signature:
        return False

    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    basestring = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(SLACK_SIGNING_SECRET.encode(), basestring.encode(), hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected, signature)


### Historical backtesting
def fetch_historical_messages(limit: int = 200) -> Iterable[dict]:
    resp = slack_client.conversations_history(
        channel=SLACK_CHANNEL_ID,
        limit=limit,
    )

    for msg in resp.get("messages", []):
        if msg.get("subtype"):
            continue
        if msg.get("thread_ts"):
            continue
        if not msg.get("text"):
            continue
        yield msg


def backtest(limit: int = 50):
    print(f"Backtesting last {limit} messages\n")

    for msg in fetch_historical_messages(limit):
        text = msg.get["text"]
        result = classify_message(text)

        print("-" * 60)
        print(text)
        print(f"→ {result.label} ({result.confidence:.2f})")
        print(f"Reason: {result.reason}")


### FastAPI app

app = FastAPI()


@app.post("/slack/events")
async def slack_events(req: Request):
    body = await req.body()
    if not verify_slack_request(req, body):
        return {"error": "Invalid request"}

    payload = await req.json()

    # Slack URL verification
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event", {})

    # Aggressive filtering (save, non-invasive)
    if event.get("type") != "message":
        return {"ok": True}
    if event.get("channel") != SLACK_CHANNEL_ID:
        return {"ok": True}
    if event.get("subtype"):
        return {"ok": True}
    if event.get("thread_ts"):
        return {"ok": True}

    text = event.get("text", "").strip()
    if not text:
        return {"ok": True}

    result = classify_message(text)

    print("\nLIVE EVENT")
    print(text)
    print(result)

    if not DRY_RUN:
        slack_client.chat_postMessage(
            channel=event["channel"],
            thread_ts=event["ts"],
            text=f"🧠 Lead classification: *{result.label}* ({result.confidence:.2f})",
        )

    return {"ok": True}


### Entrypoint
if __name__ == "__main__":
    backtest(limit=20)
