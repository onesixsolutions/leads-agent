import logfire
from rich import print

from leads_agent.config import get_settings


def configure_logfire():
    settings = get_settings()
    logfire_token = settings.logfire_token.get_secret_value() if settings.logfire_token else None
    if not logfire_token:
        print("[red]LOGFIRE_TOKEN is not set[/]")
        return

    logfire.configure(token=logfire_token)
    logfire.instrument_pydantic_ai()
