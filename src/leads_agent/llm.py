from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .config import Settings
from .domain import LeadClassification


SYSTEM_PROMPT = """\
You classify inbound leads from a consulting company contact form.

Definitions:
- spam: irrelevant, automated, SEO, crypto, junk
- solicitation: vendors, sales pitches, recruiters, partnerships
- promising: genuine inquiry about services or collaboration

Rules:
- Be conservative
- If unclear, choose spam
- Provide a short reason
"""


@lru_cache(maxsize=8)
def _classifier(llm_base_url: str, llm_model_name: str) -> Agent[None, LeadClassification]:
    model = OpenAIModel(
        base_url=llm_base_url,
        api_key="local",
        model_name=llm_model_name,
    )
    return Agent(model=model, result_type=LeadClassification, system_prompt=SYSTEM_PROMPT)


def classify_message(settings: Settings, text: str) -> LeadClassification:
    agent = _classifier(settings.llm_base_url, settings.llm_model_name)
    result = agent.run_sync(
        f"""
Message:
\"\"\"
{text}
\"\"\"
"""
    )
    return result.data

