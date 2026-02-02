import json
from pathlib import Path

from pydantic import BaseModel, Field

from leads_agent.prompts.prompts import (
    BASE_RESEARCH_PROMPT,
    BASE_SCORING_PROMPT,
    BASE_SYSTEM_PROMPT,
    BASE_TRIAGE_PROMPT,
)


class ICPConfig(BaseModel):
    """Ideal Client Profile configuration."""

    description: str | None = Field(
        default=None,
        description="Free-form description of your ideal client profile",
        examples=[
            "Mid-market B2B SaaS companies looking to modernize their data infrastructure"
        ],
    )
    target_industries: list[str] | None = Field(
        default=None,
        description="List of target industries",
        examples=[["SaaS", "FinTech", "HealthTech", "E-commerce"]],
    )
    target_company_sizes: list[str] | None = Field(
        default=None,
        description="List of target company sizes",
        examples=[["Startup", "SMB", "Mid-Market"]],
    )
    target_roles: list[str] | None = Field(
        default=None,
        description="Target roles/titles for decision makers",
        examples=[["CTO", "VP Engineering", "Head of Data", "Technical Founder"]],
    )
    geographic_focus: list[str] | None = Field(
        default=None,
        description="Geographic regions of interest",
        examples=[["US", "Canada", "UK", "EU"]],
    )
    disqualifying_signals: list[str] | None = Field(
        default=None,
        description="Signals that indicate a lead is not a good fit",
        examples=[["Requesting free services", "Student projects", "Personal use"]],
    )


class PromptConfig(BaseModel):
    """
    Deployment-specific prompt configuration.

    All fields are optional - only configured fields will be added to the prompt.
    """

    # Company/service description
    company_name: str | None = Field(
        default=None,
        description="Your company name (used in prompts)",
    )
    services_description: str | None = Field(
        default=None,
        description="Brief description of your services",
        examples=["AI/ML consulting and custom software development"],
    )

    # ICP Configuration
    icp: ICPConfig | None = Field(
        default=None,
        description="Ideal Client Profile configuration",
    )

    # Custom qualifying questions
    qualifying_questions: list[str] | None = Field(
        default=None,
        description="Questions to consider when evaluating leads",
        examples=[
            [
                "Does this look like a real business need vs. a student project?",
                "Is there budget indication or enterprise context?",
                "Is the request aligned with our core services?",
            ]
        ],
    )

    # Additional custom instructions
    custom_instructions: str | None = Field(
        default=None,
        description="Additional instructions to append to the system prompt",
    )

    # Research-specific configuration
    research_focus_areas: list[str] | None = Field(
        default=None,
        description="Specific areas to focus on during lead research",
        examples=[
            ["Technical stack", "Recent funding", "Team size", "Current challenges"]
        ],
    )

    def is_empty(self) -> bool:
        """Check if configuration has any values set."""
        return all(
            v is None
            for v in [
                self.company_name,
                self.services_description,
                self.icp,
                self.qualifying_questions,
                self.custom_instructions,
                self.research_focus_areas,
            ]
        )


class PromptManager:
    """
    Manages prompt configuration and builds dynamic prompts.

    Loads configuration from (in priority order):
    1. Runtime configuration (set via API)
    2. Environment variable PROMPT_CONFIG_JSON
    3. Config file (prompt_config.json in project root)
    4. Defaults (empty configuration)
    """

    def __init__(self, config: PromptConfig | None = None):
        self._config = config or PromptConfig()
        self._runtime_config: PromptConfig | None = None

    @property
    def config(self) -> PromptConfig:
        """Get the effective configuration (runtime overrides base)."""
        if self._runtime_config is not None:
            return self._runtime_config
        return self._config

    def update_config(self, config: PromptConfig) -> None:
        """Update runtime configuration."""
        self._runtime_config = config

    def reset_config(self) -> None:
        """Reset to base configuration (clear runtime overrides)."""
        self._runtime_config = None

    def build_classification_prompt(self) -> str:
        """
        Build the complete classification system prompt.

        Combines base prompt with deployment-specific configuration.
        """
        parts = [BASE_SYSTEM_PROMPT]
        cfg = self.config

        # Add company context
        if cfg.company_name or cfg.services_description:
            context_parts = []
            if cfg.company_name:
                context_parts.append(f"Company: {cfg.company_name}")
            if cfg.services_description:
                context_parts.append(f"Services: {cfg.services_description}")
            parts.append(
                "\n--- Internal Company Context ---\n" + "\n".join(context_parts)
            )

        # Add ICP criteria
        if cfg.icp and not cfg.icp.model_dump(exclude_none=True) == {}:
            icp = cfg.icp
            icp_parts = []

            if icp.description:
                icp_parts.append(f"**Target Profile:** {icp.description}")

            if icp.target_industries:
                icp_parts.append(
                    f"**Target Industries:** {', '.join(icp.target_industries)}"
                )

            if icp.target_company_sizes:
                icp_parts.append(
                    f"**Target Company Sizes:** {', '.join(icp.target_company_sizes)}"
                )

            if icp.target_roles:
                icp_parts.append(
                    f"**Decision Maker Roles:** {', '.join(icp.target_roles)}"
                )

            if icp.geographic_focus:
                icp_parts.append(
                    f"**Geographic Focus:** {', '.join(icp.geographic_focus)}"
                )

            if icp.disqualifying_signals:
                icp_parts.append(
                    f"**Disqualifying Signals:** {', '.join(icp.disqualifying_signals)}"
                )

            if icp_parts:
                parts.append("\n--- Ideal Client Profile ---\n" + "\n".join(icp_parts))

        # Add qualifying questions
        if cfg.qualifying_questions:
            questions = "\n".join(f"- {q}" for q in cfg.qualifying_questions)
            parts.append(
                f"\n--- Qualifying Questions ---\nConsider these when classifying:\n{questions}"
            )

        # Add custom instructions
        if cfg.custom_instructions:
            parts.append(
                f"\n--- Additional Instructions ---\n{cfg.custom_instructions}"
            )

        return "\n".join(parts)

    def build_triage_prompt(self) -> str:
        """
        Build triage system prompt.

        Uses the same deployment-specific config as classification, but tuned for speed.
        """
        # Reuse the classification prompt sections (company context, ICP, questions, etc.),
        # but with a triage-focused base prompt.
        parts = [BASE_TRIAGE_PROMPT]
        cfg = self.config

        # Add company context
        if cfg.company_name or cfg.services_description:
            context_parts = []
            if cfg.company_name:
                context_parts.append(f"Company: {cfg.company_name}")
            if cfg.services_description:
                context_parts.append(f"Services: {cfg.services_description}")
            parts.append(
                "\n--- Internal Company Context ---\n" + "\n".join(context_parts)
            )

        # Add ICP criteria
        if cfg.icp and not cfg.icp.model_dump(exclude_none=True) == {}:
            icp = cfg.icp
            icp_parts = []

            if icp.description:
                icp_parts.append(f"**Target Profile:** {icp.description}")
            if icp.target_industries:
                icp_parts.append(
                    f"**Target Industries:** {', '.join(icp.target_industries)}"
                )
            if icp.target_company_sizes:
                icp_parts.append(
                    f"**Target Company Sizes:** {', '.join(icp.target_company_sizes)}"
                )
            if icp.target_roles:
                icp_parts.append(
                    f"**Decision Maker Roles:** {', '.join(icp.target_roles)}"
                )
            if icp.geographic_focus:
                icp_parts.append(
                    f"**Geographic Focus:** {', '.join(icp.geographic_focus)}"
                )
            if icp.disqualifying_signals:
                icp_parts.append(
                    f"**Disqualifying Signals:** {', '.join(icp.disqualifying_signals)}"
                )

            if icp_parts:
                parts.append("\n--- Ideal Client Profile ---\n" + "\n".join(icp_parts))

        # Add qualifying questions
        if cfg.qualifying_questions:
            questions = "\n".join(f"- {q}" for q in cfg.qualifying_questions)
            parts.append(
                f"\n--- Qualifying Questions ---\nConsider these during triage:\n{questions}"
            )

        # Add custom instructions
        if cfg.custom_instructions:
            parts.append(
                f"\n--- Additional Instructions ---\n{cfg.custom_instructions}"
            )

        return "\n".join(parts)

    def build_scoring_prompt(self) -> str:
        """Build scoring system prompt."""
        parts = [BASE_SCORING_PROMPT]
        cfg = self.config

        # Add ICP context so scoring can incorporate fit.
        if cfg.icp:
            icp = cfg.icp
            icp_parts = []

            if icp.description:
                icp_parts.append(f"**Ideal Profile:** {icp.description}")
            if icp.target_industries:
                icp_parts.append(
                    f"**Priority Industries:** {', '.join(icp.target_industries)}"
                )
            if icp.target_company_sizes:
                icp_parts.append(
                    f"**Target Company Sizes:** {', '.join(icp.target_company_sizes)}"
                )
            if icp.target_roles:
                icp_parts.append(
                    f"**Decision Maker Roles:** {', '.join(icp.target_roles)}"
                )
            if icp.disqualifying_signals:
                icp_parts.append(
                    f"**Red Flags:** {', '.join(icp.disqualifying_signals)}"
                )

            if icp_parts:
                parts.append("\n--- Ideal Client Profile ---\n" + "\n".join(icp_parts))

        # Add qualifying questions (what matters for prioritization)
        if cfg.qualifying_questions:
            questions = "\n".join(f"- {q}" for q in cfg.qualifying_questions)
            parts.append(
                f"\n--- Qualifying Questions ---\nUse these to justify score/action:\n{questions}"
            )

        return "\n".join(parts)

    def build_research_prompt(self) -> str:
        """
        Build the complete research system prompt.

        Combines base research prompt with deployment-specific focus areas.
        """
        parts = [BASE_RESEARCH_PROMPT]
        cfg = self.config

        # Add research focus areas - what specific information to gather
        if cfg.research_focus_areas:
            areas = "\n".join(f"- {area}" for area in cfg.research_focus_areas)
            parts.append(f"\n--- What to Research ---\nFocus on finding:\n{areas}")

        # Add qualifying questions - what we're trying to determine
        if cfg.qualifying_questions:
            questions = "\n".join(f"- {q}" for q in cfg.qualifying_questions)
            parts.append(
                f"\n--- Questions to Answer ---\nTry to gather information that helps answer:\n{questions}"
            )

        # Add ICP context - what makes a lead valuable to us
        if cfg.icp:
            icp = cfg.icp
            icp_parts = []

            if icp.description:
                icp_parts.append(f"**Ideal Profile:** {icp.description}")

            if icp.target_industries:
                icp_parts.append(
                    f"**Priority Industries:** {', '.join(icp.target_industries)}"
                )

            if icp.target_company_sizes:
                icp_parts.append(
                    f"**Target Company Sizes:** {', '.join(icp.target_company_sizes)}"
                )

            if icp.target_roles:
                icp_parts.append(
                    f"**Decision Maker Roles:** {', '.join(icp.target_roles)}"
                )

            if icp.disqualifying_signals:
                icp_parts.append(
                    f"**Red Flags:** {', '.join(icp.disqualifying_signals)}"
                )

            if icp_parts:
                parts.append(
                    "\n--- Ideal Client Profile ---\nUse this context to assess fit:\n"
                    + "\n".join(icp_parts)
                )

        # Add a concrete operator clause pack derived from prompt_config to improve query quality
        clause_pack_lines: list[str] = []
        clause_pack_lines.append(
            "General noise filters: -jobs -careers -hiring -pdf -login"
        )

        if cfg.icp:
            icp = cfg.icp
            if icp.target_industries:
                industries = " OR ".join(f'"{x}"' for x in icp.target_industries)
                clause_pack_lines.append(f"Industry clause: ({industries})")
            if icp.target_roles:
                roles = " OR ".join(f'"{x}"' for x in icp.target_roles)
                clause_pack_lines.append(f"Role/title clause: ({roles})")
            if icp.geographic_focus:
                geos = " OR ".join(f'"{x}"' for x in icp.geographic_focus)
                clause_pack_lines.append(f"Geo clause: ({geos})")
            if icp.target_company_sizes:
                sizes = " OR ".join(f'"{x}"' for x in icp.target_company_sizes)
                clause_pack_lines.append(f"Company size clause: ({sizes})")
            if icp.disqualifying_signals:
                # Treat as exclusions the model can optionally apply to avoid junk
                exclusions = " ".join(f'-"{x}"' for x in icp.disqualifying_signals)
                clause_pack_lines.append(
                    f"Disqualifier exclusions (optional): {exclusions}"
                )

        if cfg.research_focus_areas:
            focus = " OR ".join(f'"{x}"' for x in cfg.research_focus_areas)
            clause_pack_lines.append(f"Focus-area terms (optional): ({focus})")

        if cfg.qualifying_questions:
            clause_pack_lines.append(
                "Qualifying questions: convert 1–2 into query clauses (e.g., pricing/budget, SOC2/compliance, headcount/employees)."
            )

        if clause_pack_lines:
            clause_pack = "\n".join(f"- {line}" for line in clause_pack_lines)
            parts.append(
                "\n--- Query Operator Clause Pack (use in DuckDuckGo queries) ---\n"
                "Use these to make searches specific. Combine with quoted company/contact names and site: constraints when useful:\n"
                f"{clause_pack}"
            )

        return "\n".join(parts)


def load_prompt_config_from_file(path: Path | str | None = None) -> PromptConfig:
    """
    Load prompt configuration from JSON file.

    Args:
        path: Explicit path to config file. If None, searches in order:
              1. PROMPT_CONFIG_PATH environment variable
              2. prompt_config.json in current directory
              3. config/prompt_config.json
    """
    import os

    if path is None:
        # Check environment variable first
        env_path = os.environ.get("PROMPT_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if not path.is_file():
                print(f"[WARN] PROMPT_CONFIG_PATH set but file not found: {env_path}")
                return PromptConfig()
        else:
            # Search for config file in default locations
            candidates = [
                Path("prompt_config.json"),
                Path("config/prompt_config.json"),
                Path.cwd() / "prompt_config.json",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    path = candidate
                    break

    if path is None:
        return PromptConfig()

    path = Path(path)
    if not path.is_file():
        return PromptConfig()

    try:
        data = json.loads(path.read_text())
        return PromptConfig.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] Failed to load prompt config from {path}: {e}")
        return PromptConfig()


def load_prompt_config() -> PromptConfig:
    """
    Load prompt configuration from available sources.

    Priority:
    1. File specified by PROMPT_CONFIG_PATH environment variable
    2. prompt_config.json in current directory
    3. config/prompt_config.json
    4. Empty defaults
    """
    return load_prompt_config_from_file()


# Global prompt manager instance
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get or create the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        config = load_prompt_config()
        _prompt_manager = PromptManager(config)
    return _prompt_manager


def reset_prompt_manager() -> None:
    """Reset the global prompt manager (useful for testing)."""
    global _prompt_manager
    _prompt_manager = None
