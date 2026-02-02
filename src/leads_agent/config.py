from pathlib import Path

import typer
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from leads_agent.common import mask_secret

console = Console()


def _find_dotenv() -> Path | None:
    """Search for .env file from cwd upward to find project root."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
        # Stop at common project root indicators
        if (parent / "pyproject.toml").is_file() or (parent / ".git").is_dir():
            # Check one more time in case .env is here
            if candidate.is_file():
                return candidate
            break
    return None


class Settings(BaseSettings):
    """
    Runtime configuration.

    Values are loaded from environment variables and `.env` (if present).
    Searches for .env from current directory upward to project root.
    """

    model_config = SettingsConfigDict(
        env_file=_find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Slack
    slack_bot_token: SecretStr | None = Field(
        default=None, validation_alias="SLACK_BOT_TOKEN"
    )
    slack_app_token: SecretStr | None = Field(
        default=None, validation_alias="SLACK_APP_TOKEN"
    )
    slack_channel_id: str | None = Field(
        default=None, validation_alias="SLACK_CHANNEL_ID"
    )
    slack_test_channel_id: str | None = Field(
        default=None, validation_alias="SLACK_TEST_CHANNEL_ID"
    )

    # LLM (OpenAI by default; works with any OpenAI-compatible API)
    llm_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL"
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini", validation_alias="LLM_MODEL_NAME"
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )

    # Behavior
    dry_run: bool = Field(default=True, validation_alias="DRY_RUN")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Note: Prompt configuration is handled separately via PROMPT_CONFIG_PATH env var
    # or auto-discovered prompt_config.json file. See leads_agent.prompts module.

    def require_slack_socket_mode(self) -> "Settings":
        """Validate settings required for Socket Mode."""
        missing: list[str] = []
        if self.slack_bot_token is None:
            missing.append("SLACK_BOT_TOKEN")
        if self.slack_app_token is None:
            missing.append("SLACK_APP_TOKEN")
        if missing:
            raise ValueError(f"Missing required Slack config: {', '.join(missing)}")
        return self

    def require_slack_client(self) -> "Settings":
        """Validate settings required for Slack API calls (backtest, test, etc.)."""
        missing: list[str] = []
        if self.slack_bot_token is None:
            missing.append("SLACK_BOT_TOKEN")
        if missing:
            raise ValueError(f"Missing required Slack config: {', '.join(missing)}")
        return self


def get_settings() -> Settings:
    """Get settings instance (convenience for CLI)."""
    return Settings()


def _find_prompt_config_source() -> str | None:
    """Find where prompt configuration is being loaded from."""
    import os

    # Check env var first
    env_path = os.environ.get("PROMPT_CONFIG_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    # Check default locations
    candidates = [
        Path("prompt_config.json"),
        Path("config/prompt_config.json"),
        Path.cwd() / "prompt_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def display_config():
    try:
        settings = get_settings()
    except Exception as e:
        rprint(f"[red]Error loading settings:[/] {e}")
        raise typer.Exit(1)

    table = Table(
        title="Current Configuration", show_header=True, header_style="bold cyan"
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("SLACK_BOT_TOKEN", mask_secret(settings.slack_bot_token))
    table.add_row("SLACK_APP_TOKEN", mask_secret(settings.slack_app_token))
    table.add_row("SLACK_CHANNEL_ID", settings.slack_channel_id or "[not set]")
    table.add_row(
        "SLACK_TEST_CHANNEL_ID", settings.slack_test_channel_id or "[not set]"
    )
    table.add_row("OPENAI_API_KEY", mask_secret(settings.openai_api_key))
    table.add_row("LLM_BASE_URL", settings.llm_base_url)
    table.add_row("LLM_MODEL_NAME", settings.llm_model_name)
    table.add_row("DRY_RUN", str(settings.dry_run))
    table.add_row("DEBUG", str(settings.debug))

    # Show prompt config path
    prompt_config_source = _find_prompt_config_source()
    table.add_row("PROMPT_CONFIG", prompt_config_source or "[default]")

    console.print(table)

    if prompt_config_source:
        rprint(
            "\n[dim]Run [bold]leads-agent prompts[/] to view prompt configuration[/]"
        )
