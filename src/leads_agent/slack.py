from slack_sdk import WebClient

from leads_agent.config import Settings


def slack_client(settings: Settings) -> WebClient:
    """Create a Slack WebClient instance."""
    token = settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else None
    return WebClient(token=token)
