import httpx
import json
from app.core.config import settings
from app.services.ml.types import PipelineStepResult


SYSTEM_PROMPT = """You are an ML teaching assistant embedded in an interactive machine learning playground.
Your job is to explain what just happened after a student applies a technique to their dataset.
Rules:
- Be specific — always reference the actual numbers from the stats provided
- Keep it to 3-5 sentences
- Use plain English, avoid unnecessary jargon
- If something unexpected happened, point it out
- Never suggest other techniques unless asked
- Never repeat what the technique is in general terms — focus on what happened THIS time with THIS data"""


def build_prompt(result: PipelineStepResult, dataset_name: str) -> str:
    return f"""The student is working on the {dataset_name} dataset.

They just applied this step:
- Step: {result.step}
- Technique: {result.technique}
- Parameters: {json.dumps(result.params)}

What changed in the data:
{json.dumps(result.stats, indent=2)}

Warnings raised:
{result.warnings if result.warnings else "none"}

Metrics impact (if available):
{json.dumps(result.metrics_delta) if result.metrics_delta else "not yet evaluated — training hasn't run yet"}

Explain what happened and why, referencing the specific numbers above."""


async def get_explanation(result: PipelineStepResult, dataset_name: str) -> str:
    prompt = build_prompt(result, dataset_name)

    if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        return await _call_anthropic(prompt)
    elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return await _call_openai(prompt)
    else:
        return _fallback_explanation(result)


async def _call_anthropic(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = response.json()
        return data["content"][0]["text"]


async def _call_openai(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]


def _fallback_explanation(result: PipelineStepResult) -> str:
    """Used when no API key is configured — rule-based fallback."""
    stats = result.stats
    warnings_text = " ".join(result.warnings) if result.warnings else ""

    if result.step == "missing_values":
        before = stats.get("missing_before", 0)
        after = stats.get("missing_after", 0)
        return (
            f"Applied {result.technique} imputation. "
            f"Missing values reduced from {before} to {after}. "
            f"{warnings_text}"
        ).strip()

    if result.step == "scaling":
        cols = stats.get("n_columns_scaled", 0)
        return (
            f"Applied {result.technique} scaling to {cols} numeric columns. "
            f"Features are now on a comparable scale. "
            f"{warnings_text}"
        ).strip()

    if result.step == "encoding":
        encoded = stats.get("encoded_columns", [])
        new_cols = stats.get("new_cols_created", 0)
        return (
            f"Encoded {len(encoded)} categorical columns using {result.technique}. "
            f"{new_cols} new columns were created. "
            f"{warnings_text}"
        ).strip()

    return f"Applied {result.technique} to step '{result.step}'. {warnings_text}".strip()