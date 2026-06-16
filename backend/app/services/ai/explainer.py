import httpx
import json
from app.core.config import settings
from app.services.ml.types import PipelineStepResult


SYSTEM_PROMPT = """You are an ML teaching assistant embedded in an interactive \
machine learning playground.

Your job is to explain what just happened after a student applies a technique \
to their dataset.

Rules:
- Be specific — always reference the actual numbers from the stats provided
- Keep it to 3-5 sentences maximum
- Use plain English, avoid unnecessary jargon
- If something unexpected happened, point it out clearly
- Never suggest other techniques unless the stats reveal a problem
- Never explain what the technique is in general — focus on what happened \
THIS time with THIS data
- Never start with "I" or "The"
- Always ground every claim in the numbers provided"""


def build_prompt(result: PipelineStepResult, dataset_name: str) -> str:
    return f"""The student is working on the {dataset_name} dataset.

They just applied this preprocessing step:
- Step      : {result.step}
- Technique : {result.technique}
- Parameters: {json.dumps(result.params)}

What changed in the data:
{json.dumps(result.stats, indent=2)}

Warnings raised:
{result.warnings if result.warnings else "none"}

Metrics impact (if model has been trained):
{json.dumps(result.metrics_delta) if result.metrics_delta else "not yet evaluated — training has not run"}

Explain what happened and why, referencing the specific numbers above. \
Do not repeat the technique name as a definition."""


async def get_explanation(result: PipelineStepResult, dataset_name: str) -> str:
    """
    Get AI explanation for a preprocessing step result.
    Tries configured LLM provider first, falls back to rule-based explanation.
    Never raises — always returns a string.
    """
    prompt = build_prompt(result, dataset_name)

    # Try LLM if key is configured
    if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        try:
            return await _call_anthropic(prompt)
        except Exception as e:
            # Log but don't crash — fall through to fallback
            print(f"[AI] Anthropic call failed: {e}. Using fallback.")

    elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            return await _call_openai(prompt)
        except Exception as e:
            print(f"[AI] OpenAI call failed: {e}. Using fallback.")

    return _rule_based_explanation(result)


async def _call_anthropic(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "system":     SYSTEM_PROMPT,
                "messages":   [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        # Validate response shape before accessing
        if "content" not in data or not data["content"]:
            raise ValueError(f"Unexpected Anthropic response shape: {data}")

        return data["content"][0]["text"].strip()


async def _call_openai(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      "gpt-4o-mini",
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Unexpected OpenAI response shape: {data}")

        return data["choices"][0]["message"]["content"].strip()


def _rule_based_explanation(result: PipelineStepResult) -> str:
    """
    Deterministic fallback when no LLM API key is configured or calls fail.
    Generates specific explanations from the stats dict — never generic.
    """
    stats = result.stats
    warnings_text = " ".join(result.warnings) if result.warnings else ""

    if result.step == "missing_values":
        before = stats.get("missing_before", 0)
        after  = stats.get("missing_after", 0)
        rows_b = stats.get("rows_before", "?")
        rows_a = stats.get("rows_after", "?")
        filled = before - after

        if result.technique == "drop_rows":
            dropped = int(rows_b) - int(rows_a) if str(rows_b).isdigit() and str(rows_a).isdigit() else "some"
            return (
                f"Dropped {dropped} rows that contained missing values. "
                f"{rows_a} rows remain. "
                f"This approach loses data but guarantees no nulls reach the model. "
                f"{warnings_text}"
            ).strip()

        if result.technique == "drop_cols":
            cols_b = stats.get("cols_before", "?")
            cols_a = stats.get("cols_after", "?")
            return (
                f"Removed columns with more than 50% missing values. "
                f"Column count reduced from {cols_b} to {cols_a}. "
                f"{warnings_text}"
            ).strip()

        per_col = stats.get("per_column_before", {})
        most_missing = max(per_col, key=lambda k: per_col[k]) if per_col else None
        most_missing_count = per_col[most_missing] if most_missing else 0

        return (
            f"Filled {filled} missing values using {result.technique} imputation. "
            f"{'Most missing were in ' + most_missing + ' (' + str(most_missing_count) + ' nulls). ' if most_missing else ''}"
            f"{after} nulls remain after treatment. "
            f"{warnings_text}"
        ).strip()

    if result.step == "outliers":
        total      = stats.get("total_outliers_found", 0)
        rows_b     = stats.get("rows_before", "?")
        rows_a     = stats.get("rows_after", "?")
        per_col    = stats.get("per_column", {})
        most_col   = max(per_col, key=lambda k: per_col[k]) if per_col else None
        most_count = per_col[most_col] if most_col else 0

        if result.technique == "iqr_cap":
            return (
                f"Found {total} outlier values across numeric columns and capped them at IQR boundaries. "
                f"{'Most outliers were in ' + most_col + ' (' + str(most_count) + '). ' if most_col else ''}"
                f"All {rows_a} rows are preserved — capping is non-destructive. "
                f"{warnings_text}"
            ).strip()

        if result.technique == "zscore_remove":
            dropped = int(str(rows_b)) - int(str(rows_a)) if str(rows_b).isdigit() and str(rows_a).isdigit() else "some"
            return (
                f"Removed {dropped} rows where any numeric feature exceeded 3 standard deviations. "
                f"{rows_a} rows remain. "
                f"{total} individual outlier values were detected. "
                f"{warnings_text}"
            ).strip()

        return (
            f"{total} outliers detected using {result.technique}. "
            f"Rows before: {rows_b}, after: {rows_a}. "
            f"{warnings_text}"
        ).strip()

    if result.step == "feature_engineering":
        new   = stats.get("new_features_created", 0)
        names = stats.get("new_feature_names", [])
        cols_b = stats.get("cols_before", "?")
        cols_a = stats.get("cols_after", "?")

        if new == 0:
            return f"No new features created. Step skipped. {warnings_text}".strip()

        sample = ", ".join(names[:3])
        return (
            f"Created {new} new features using {result.technique} — "
            f"column count grew from {cols_b} to {cols_a}. "
            f"Example new features: {sample}. "
            f"More features can improve recall but risk overfitting — "
            f"feature selection in the next step will prune low-value ones. "
            f"{warnings_text}"
        ).strip()

    if result.step == "encoding":
        encoded = stats.get("encoded_columns", [])
        new     = stats.get("new_cols_created", 0)
        cols_a  = stats.get("cols_after", "?")

        if not encoded:
            return (
                "No categorical columns found — encoding step had nothing to do. "
                "Dataset may already be fully numeric."
            )

        return (
            f"Encoded {len(encoded)} categorical column(s) using {result.technique}. "
            f"{'One-hot encoding created ' + str(new) + ' new binary columns. ' if result.technique == 'onehot' and new > 0 else ''}"
            f"Dataset now has {cols_a} total columns. "
            f"{warnings_text}"
        ).strip()

    if result.step == "feature_selection":
        dropped  = stats.get("n_dropped", 0)
        cols_b   = stats.get("cols_before", "?")
        cols_a   = stats.get("cols_after", "?")
        drop_list = stats.get("dropped_columns", [])

        if dropped == 0:
            return (
                f"No features were dropped by {result.technique}. "
                f"All {cols_b} features passed the selection criteria. "
                f"{warnings_text}"
            ).strip()

        sample = ", ".join(drop_list[:3]) if drop_list else "none listed"
        return (
            f"Dropped {dropped} feature(s) using {result.technique}, "
            f"reducing from {cols_b} to {cols_a} columns. "
            f"Dropped: {sample}{'...' if len(drop_list) > 3 else ''}. "
            f"Fewer, more relevant features often improve generalisation. "
            f"{warnings_text}"
        ).strip()

    if result.step == "scaling":
        n      = stats.get("n_columns_scaled", 0)
        sample = stats.get("sample_before", {})
        first_col = next(iter(sample), None)

        if n == 0:
            return f"No scaling applied. {warnings_text}".strip()

        before_mean = sample[first_col]["mean"] if first_col else "?"
        after_sample = stats.get("sample_after", {})
        after_mean   = after_sample[first_col]["mean"] if first_col else "?"

        return (
            f"Scaled {n} numeric columns using {result.technique}. "
            f"{'For example, ' + first_col + ' mean changed from ' + str(before_mean) + ' to ' + str(after_mean) + '. ' if first_col else ''}"
            f"Scaling is fit on training data only during model training — "
            f"this display uses the full dataset for illustration. "
            f"{warnings_text}"
        ).strip()

    return (
        f"Applied {result.technique} to step '{result.step}'. "
        f"{warnings_text}"
    ).strip()