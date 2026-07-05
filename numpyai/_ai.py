"""LLM-driven NumPy code generation via Pydantic AI."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import Agent

T = TypeVar("T")

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_lock = threading.Lock()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return a module-wide, long-lived event loop running in a daemon thread.

    Reused across all calls so that clients (httpx, etc.) held by Pydantic AI
    agents stay bound to a live loop between successive ``.chat()`` calls.
    """
    global _worker_loop
    with _worker_lock:
        if _worker_loop is None or _worker_loop.is_closed():
            _worker_loop = asyncio.new_event_loop()
            threading.Thread(
                target=_worker_loop.run_forever,
                daemon=True,
                name="numpyai-asyncio",
            ).start()
        return _worker_loop


def _run_coro(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code.

    Works in scripts, in Jupyter (where an event loop is already running), and
    across multiple sequential calls without invalidating async HTTP clients.
    """
    fut = asyncio.run_coroutine_threadsafe(coro, _get_worker_loop())  # type: ignore[arg-type]
    return fut.result()

DEFAULT_MODEL = "google:gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a coding assistant who generates only NumPy and Python code. "
    "You respond with executable Python that operates on pre-defined arrays."
)


class CodeResponse(BaseModel):
    """Structured response returned by the code-generation agent."""

    code: str = Field(
        description=(
            "Executable Python/NumPy code. Must define `output` (the result) and "
            "`metadata` (a short string describing `output`). No markdown fences."
        )
    )
    explanation: str = Field(
        description="One-sentence natural-language explanation of what the code does."
    )


class Judgment(BaseModel):
    """Independent judgment of whether generated code answers the user's query.

    This is a *classification* task, not a code-generation task - so it cannot
    fail with syntax errors, missing names, or misapplied math (all failure
    modes of the previous LLM-rewrites-verification-code design).
    """

    interprets_query_correctly: bool = Field(
        description=(
            "True iff the generated code computes what the user's query asks for. "
            "Judge intent only; do not re-derive the answer."
        )
    )
    reason: str = Field(
        description=(
            "Short reason. If False, name the specific misinterpretation "
            "(e.g. 'query asked for mean, code computes median'). If True, empty."
        ),
        default="",
    )


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial reviewer. You classify whether a short NumPy snippet "
    "correctly interprets a natural-language query. Do NOT re-derive the "
    "numerical answer. Judge only whether the code addresses what was asked."
)


class NumpyCodeGen:
    """Generates NumPy code from natural-language queries using Pydantic AI.

    Parameters
    ----------
    model:
        Any model spec accepted by ``pydantic-ai`` - for example
        ``"google:gemini-2.5-flash"``, ``"anthropic:claude-sonnet-4-5"``,
        ``"openai:gpt-4o"``, or a pre-configured ``pydantic_ai.models.Model``
        instance. Defaults to Google Gemini 2.5 Flash.
    system_prompt:
        Optional override for the agent's system prompt.
    """

    def __init__(
        self,
        model: Any = DEFAULT_MODEL,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        prompt = system_prompt or SYSTEM_PROMPT
        self._code_agent: Agent[None, CodeResponse] = Agent(
            model=model,
            output_type=CodeResponse,
            system_prompt=prompt,
        )
        self._text_agent: Agent[None, str] = Agent(
            model=model,
            system_prompt=prompt,
        )
        self._judge_agent: Agent[None, Judgment] = Agent(
            model=model,
            output_type=Judgment,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

    def generate_code(self, prompt: str) -> CodeResponse:
        """Run the code-generation agent and return structured output."""
        result = _run_coro(self._code_agent.run(prompt))
        return result.output

    def generate_text(self, prompt: str) -> str:
        """Run the free-form text agent (used by :class:`Diagnosis`)."""
        result = _run_coro(self._text_agent.run(prompt))
        return result.output

    def judge(self, query: str, code: str, metadata: str) -> Judgment:
        """Classify whether ``code`` correctly interprets ``query``.

        Structured yes/no output only - no code generation, no math.
        """
        prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"GENERATED CODE (defines `output`):\n{code}\n\n"
            f"AUTHOR'S DESCRIPTION OF `output`:\n{metadata}\n\n"
            "Question: does this code compute what the query asks for? "
            "Answer via the structured output. Do NOT recompute the answer."
        )
        return _run_coro(self._judge_agent.run(prompt)).output

    @staticmethod
    def prompt_single(
        query: str,
        metadata: dict,
        prior_feedback: str | None = None,
    ) -> str:
        feedback_block = (
            f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
            "Correct that specific issue this time.\n"
            if prior_feedback
            else ""
        )
        return f"""Generate NumPy code to perform the following operation:

{query}
{feedback_block}

CRITICAL INSTRUCTIONS:
1. The array is ALREADY defined as `arr`. DO NOT create a new array with `arr = ...`.
2. DO NOT IMPORT any libraries except numpy (already imported as `np`).
3. Prefer NumPy for everything. `sklearn` and `matplotlib.pyplot` (as `plt`) are
   available as a last resort - do not use them unless necessary.
4. Return ONLY code that operates on the existing `arr` variable.
5. There MUST be exactly one variable named `output` containing what the user asked for.
6. There MUST be exactly one variable named `metadata` - a short string describing `output`.
7. Ensure data is properly cleaned before executing any computation.

The array has these properties:
{metadata}

CORRECT EXAMPLES:
    # Replace NaN values with zero
    output = np.where(np.isnan(arr), 0, arr)
    metadata = "arr with NaNs replaced by 0"

    # Calculate mean of array
    output = np.mean(arr)
    metadata = "scalar: mean of arr"
"""

    @staticmethod
    def prompt_multiple(
        query: str,
        context: dict,
        prior_feedback: str | None = None,
    ) -> str:
        def format_metadata(md: dict) -> str:
            parts = [
                f"Shape: {md['shape']}, Dims: {md['dims']}, Type: {md['element_type']}",
                f"Size: {md['size']} elements, Memory: {md['byte_size']} bytes",
            ]
            if md.get("has_nan"):
                parts.append("Contains NaN values")
            if md.get("has_inf"):
                parts.append("Contains infinite values")
            if "min" in md and "max" in md:
                parts.append(f"Range: [{md['min']}, {md['max']}]")
            if "zeros_count" in md and "non_zeros_count" in md:
                parts.append(
                    f"Zero elements: {md['zeros_count']}, "
                    f"Non-zero elements: {md['non_zeros_count']}"
                )
            return "; ".join(parts)

        array_descriptions = "\n".join(
            f"- **{name}**: {format_metadata(info['metadata'])}"
            for name, info in context.items()
        )

        names = ", ".join(context.keys())
        feedback_block = (
            f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
            "Correct that specific issue this time.\n"
            if prior_feedback
            else ""
        )
        return f"""Generate NumPy code to perform the following operation:

{query}
{feedback_block}

CRITICAL INSTRUCTIONS:
1. These arrays are ALREADY defined: {names}. DO NOT redefine them.
2. DO NOT IMPORT any libraries except numpy (already imported as `np`).
3. Prefer NumPy. `sklearn` and `matplotlib.pyplot` (as `plt`) are available only as a
   last resort.
4. There MUST be exactly one variable named `output` containing the result of the query.
5. There MUST be exactly one variable named `metadata` - a short string describing `output`.
6. Ensure data is properly cleaned before executing any computation.

Array information:
{array_descriptions}

CORRECT EXAMPLES:
    output = np.mean(arr1) - np.mean(arr2)
    metadata = "scalar: difference of the two means"

    arr2_imputed = np.where(np.isnan(arr2), np.mean(arr1), arr2)
    output = arr2_imputed
    metadata = "arr2 with NaNs replaced by mean of arr1"
"""
