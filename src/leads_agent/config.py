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
    slack_bot_token: SecretStr | None = Field(default=None, validation_alias="SLACK_BOT_TOKEN")
    slack_app_token: SecretStr | None = Field(default=None, validation_alias="SLACK_APP_TOKEN")
    slack_channel_id: str | None = Field(default=None, validation_alias="SLACK_CHANNEL_ID")
    slack_test_channel_id: str | None = Field(default=None, validation_alias="SLACK_TEST_CHANNEL_ID")

    # LLM (Anthropic Claude)
    llm_model_name: str = Field(default="claude-opus-5", validation_alias="LLM_MODEL_NAME")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_max_tokens: int = Field(default=16000, validation_alias="LLM_MAX_TOKENS")

    # Web search (research stage)
    # `ddgs` fronts several engines. The duckduckgo backend rate-limits to the
    # point of timing out from a server IP, so it is last in the chain.
    search_backends: str = Field(
        default="bing,yahoo,google,brave,mojeek,duckduckgo",
        validation_alias="SEARCH_BACKENDS",
        description="Comma-separated search engines, tried in order.",
    )
    search_timeout_s: int = Field(
        default=20,
        validation_alias="SEARCH_TIMEOUT_S",
        description="Per-request search timeout; ddgs defaults to 5s, which is too short.",
    )

    # Lead briefs (HTML brief in S3, served over HTTP)
    briefs_enabled: bool = Field(
        default=False,
        validation_alias="BRIEFS_ENABLED",
        description="Master switch. Off by default so an unconfigured deploy behaves exactly as before.",
    )
    briefs_s3_bucket: str | None = Field(
        default=None,
        validation_alias="BRIEFS_S3_BUCKET",
        description="Bucket the briefs are written to. Credentials come from the default boto3 chain, never from here.",
    )
    briefs_s3_prefix: str = Field(
        default="briefs",
        validation_alias="BRIEFS_S3_PREFIX",
        description="Key prefix inside the bucket, so a shared bucket can hold other things.",
    )
    briefs_s3_region: str | None = Field(
        default=None,
        validation_alias="BRIEFS_S3_REGION",
        description="Region for the S3 client; falls back to the boto3/AWS default when unset.",
    )
    briefs_base_url: str | None = Field(
        default=None,
        validation_alias="BRIEFS_BASE_URL",
        description="Public origin the briefs are reachable at, e.g. http://100.79.160.6:8080. Required for links to be clickable from Slack.",
    )
    briefs_http_enabled: bool = Field(
        default=True,
        validation_alias="BRIEFS_HTTP_ENABLED",
        description="Whether to run the brief HTTP listener. Off means publish-only (something else serves the bucket).",
    )
    briefs_http_host: str = Field(
        default="0.0.0.0",
        validation_alias="BRIEFS_HTTP_HOST",
        description="Bind address inside the container. Exposure is decided by which host address the port is published on, not here.",
    )
    briefs_http_port: int = Field(
        default=8080,
        validation_alias="BRIEFS_HTTP_PORT",
        description="Listener port. 80 and 5678 are taken by other containers on the current host.",
    )

    # Observability
    logfire_token: SecretStr | None = Field(default=None, validation_alias="LOGFIRE_TOKEN")

    # Behavior
    dry_run: bool = Field(default=True, validation_alias="DRY_RUN")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Note: Prompt configuration is handled separately via PROMPT_CONFIG_PATH env var
    # or auto-discovered prompt_config.json file. See leads_agent.prompts module.

    @property
    def search_backend_list(self) -> tuple[str, ...]:
        """`search_backends` parsed into an ordered tuple of engine names."""
        return tuple(b.strip() for b in self.search_backends.split(",") if b.strip())

    def briefs_effective_base_url(self) -> str | None:
        """
        The origin to build brief links from.

        Prefers the explicitly configured `BRIEFS_BASE_URL`, because the bind
        address inside a container says nothing about how the outside world
        reaches it. Falls back to host:port only when the bind host is a
        concrete address — a wildcard bind (`0.0.0.0`) cannot be turned into a
        URL anyone else can use.
        """
        if self.briefs_base_url:
            return self.briefs_base_url.rstrip("/")
        host = (self.briefs_http_host or "").strip()
        if not host or host in ("0.0.0.0", "::", "*"):
            return None
        bracketed = f"[{host}]" if ":" in host else host
        return f"http://{bracketed}:{self.briefs_http_port}"

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

    table = Table(title="Current Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("SLACK_BOT_TOKEN", mask_secret(settings.slack_bot_token))
    table.add_row("SLACK_APP_TOKEN", mask_secret(settings.slack_app_token))
    table.add_row("SLACK_CHANNEL_ID", settings.slack_channel_id or "[not set]")
    table.add_row("SLACK_TEST_CHANNEL_ID", settings.slack_test_channel_id or "[not set]")
    table.add_row("ANTHROPIC_API_KEY", mask_secret(settings.anthropic_api_key))
    table.add_row("LLM_MODEL_NAME", settings.llm_model_name)
    table.add_row("LLM_MAX_TOKENS", str(settings.llm_max_tokens))
    table.add_row("SEARCH_BACKENDS", settings.search_backends)
    table.add_row("SEARCH_TIMEOUT_S", str(settings.search_timeout_s))
    table.add_row("BRIEFS_ENABLED", str(settings.briefs_enabled))
    # Square brackets are rich markup, so these fallbacks use parentheses.
    table.add_row("BRIEFS_S3_BUCKET", settings.briefs_s3_bucket or "(not set)")
    table.add_row("BRIEFS_S3_PREFIX", settings.briefs_s3_prefix)
    table.add_row("BRIEFS_S3_REGION", settings.briefs_s3_region or "(boto3 default)")
    table.add_row(
        "BRIEFS_BASE_URL",
        settings.briefs_base_url
        or f"(derived: {settings.briefs_effective_base_url() or 'none - links will be relative'})",
    )
    table.add_row("BRIEFS_HTTP_ENABLED", str(settings.briefs_http_enabled))
    table.add_row("BRIEFS_HTTP_HOST", settings.briefs_http_host)
    table.add_row("BRIEFS_HTTP_PORT", str(settings.briefs_http_port))
    table.add_row("LOGFIRE_TOKEN", mask_secret(settings.logfire_token))
    table.add_row("DRY_RUN", str(settings.dry_run))
    table.add_row("DEBUG", str(settings.debug))

    # Show prompt config path
    prompt_config_source = _find_prompt_config_source()
    table.add_row("PROMPT_CONFIG", prompt_config_source or "[default]")

    console.print(table)

    if prompt_config_source:
        rprint("\n[dim]Run [bold]leads-agent prompts[/] to view prompt configuration[/]")
