from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime configuration.

    Values are loaded from environment variables and `.env` (if present).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Slack
    slack_bot_token: SecretStr | None = Field(default=None, validation_alias="SLACK_BOT_TOKEN")
    slack_signing_secret: SecretStr | None = Field(default=None, validation_alias="SLACK_SIGNING_SECRET")
    slack_channel_id: str | None = Field(default=None, validation_alias="SLACK_CHANNEL_ID")

    # LLM (OpenAI-compatible API; defaults work for Ollama)
    llm_base_url: str = Field(default="http://localhost:11434/v1", validation_alias="LLM_BASE_URL")
    llm_model_name: str = Field(default="llama3.1:8b", validation_alias="LLM_MODEL_NAME")

    # Behavior
    dry_run: bool = Field(default=True, validation_alias="DRY_RUN")

    def require_slack(self) -> "Settings":
        missing: list[str] = []
        if self.slack_bot_token is None:
            missing.append("SLACK_BOT_TOKEN")
        if self.slack_signing_secret is None:
            missing.append("SLACK_SIGNING_SECRET")
        if self.slack_channel_id is None:
            missing.append("SLACK_CHANNEL_ID")
        if missing:
            raise ValueError(f"Missing required Slack config: {', '.join(missing)}")
        return self


def get_settings() -> Settings:
    """Get settings instance (convenience for CLI)."""
    return Settings()
