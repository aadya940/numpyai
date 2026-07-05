"""Multi-array session for chatting over several NumPy arrays at once."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ._ai import DEFAULT_MODEL, CodeResponse, NumpyCodeGen
from ._array import array
from ._exceptions import NumpyAIError
from ._utils import NumpyMetadataCollector, clean_code
from ._validator import NumpyValidator

console = Console()


class NumpyAISession:
    """Chat with multiple NumPy arrays in a single session.

    Parameters
    ----------
    data:
        List of ``numpy.ndarray`` (or ``numpyai.array``) objects. They will be
        exposed to the LLM as ``arr1``, ``arr2``, ...
    verbose:
        Show all intermediate LLM steps.
    model:
        Any model spec accepted by ``pydantic-ai`` (default:
        ``"google:gemini-2.5-flash"``).
    max_tries:
        Number of code-generation attempts before giving up (default: 3).
    """

    def __init__(
        self,
        data: list[np.ndarray | array],
        *,
        verbose: bool = False,
        model: Any = DEFAULT_MODEL,
        max_tries: int = 3,
    ) -> None:
        self._context: dict[str, dict[str, Any]] = {}
        self._metadata_collector = NumpyMetadataCollector()
        self._code_generator = NumpyCodeGen(model=model)
        self._validator = NumpyValidator()

        self._initialize_arrays(data)
        self.current_prompt: str | None = None
        self._output_metadata: dict = {}
        self.verbose = verbose
        self.MAX_TRIES = max_tries
        self._model = model

    def _initialize_arrays(self, data: list[np.ndarray | array]) -> None:
        for i, arr in enumerate(data, start=1):
            if isinstance(arr, array):
                arr = arr.data
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"session data must be numpy.ndarray or numpyai.array, "
                    f"got {type(arr).__name__} at index {i - 1}"
                )
            self._context[f"arr{i}"] = {
                "array": arr,
                "metadata": self._metadata_collector.metadata(arr),
            }

    def chat(self, query: str) -> Any:
        """Handle a natural-language query across the session's arrays."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue"))

        error_messages: list[str] = []
        prior_feedback: str | None = None
        for attempt in range(1, self.MAX_TRIES + 1):
            is_last = attempt == self.MAX_TRIES
            if self.verbose:
                console.print(
                    Panel(
                        f"[bold green]Attempt {attempt}/{self.MAX_TRIES}...[/bold green]",
                        border_style="yellow",
                    )
                )

            try:
                self.current_prompt = query
                response = self._generate_response(query, prior_feedback)
                result, explainer = self._execute(response.code)

                if result is None:
                    error_messages.append(f"Attempt {attempt}: execution returned None.")
                    prior_feedback = "code execution produced no `output` variable"
                    if self.verbose:
                        console.print(
                            Panel(
                                "[bold red]✗[/bold red] Execution returned None.",
                                border_style="red",
                            )
                        )
                    continue

                judgment = self._code_generator.judge(
                    query=query, code=response.code, metadata=str(explainer or "")
                )
                if self.verbose or is_last:
                    self._print_judgment(judgment)

                if judgment.interprets_query_correctly:
                    if self.verbose or is_last:
                        preview = result if not isinstance(result, np.ndarray) else type(result)
                        console.print(
                            Panel(f"[bold green]Output\n {preview}", border_style="yellow")
                        )
                    return result

                prior_feedback = f"judgment rejected: {judgment.reason}"
                error_messages.append(f"Attempt {attempt}: {prior_feedback}")

            except Exception as e:
                error_messages.append(f"Attempt {attempt}: {e}")
                prior_feedback = f"exception in previous attempt: {e}"
                if self.verbose or is_last:
                    console.print(
                        Panel(
                            f"[bold red]✗[/bold red] Attempt {attempt} failed: {e}",
                            border_style="red",
                        )
                    )

        self._print_error_table(error_messages)
        warnings.warn(
            f"Validation failed after {self.MAX_TRIES} attempts. Please check the validity of the code.",
            stacklevel=2,
        )
        return None

    def _generate_response(
        self, query: str, prior_feedback: str | None = None
    ) -> CodeResponse:
        prompt = self._code_generator.prompt_multiple(
            query=query, context=self._context, prior_feedback=prior_feedback
        )
        response = self._code_generator.generate_code(prompt)
        response = CodeResponse(
            code=clean_code(response.code),
            explanation=response.explanation,
        )

        if self.verbose:
            console.print(
                Panel(
                    Syntax(response.code, "python", theme="monokai", line_numbers=True),
                    title="[bold]Generated Code[/bold]",
                    border_style="blue",
                )
            )

        if not self._validator.validate_code(response.code):
            raise NumpyAIError("Generated code failed syntax validation")
        return response

    @staticmethod
    def _print_judgment(j) -> None:
        mark = "[green]✓[/green]" if j.interprets_query_correctly else "[red]✗[/red]"
        body = j.reason or "correctly interprets the query"
        console.print(Panel(f"{mark} {body}", title="Judgment", border_style="magenta"))

    def _execute(self, code: str, code_out: Any = None) -> tuple[Any, Any]:
        """Execute generated code in a controlled namespace.

        Returns ``(result, explainer)``. On error, returns ``(None, None)``.
        """
        local_vars: dict[str, Any] = {"np": np}
        for name, info in self._context.items():
            local_vars[name] = info["array"]
        if code_out is not None:
            local_vars["code_out"] = code_out

        exec_globals: dict[str, Any] = {"__builtins__": __builtins__, "np": np}
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            exec_globals["plt"] = plt
        except ImportError:
            pass
        try:
            import sklearn  # noqa: F401
            exec_globals["sklearn"] = sklearn
        except ImportError:
            pass

        try:
            exec(code, exec_globals, local_vars)
            result = local_vars.get("output")
            explainer = local_vars.get("metadata")

            if result is not None and self.verbose:
                lines = str(result).split("\n")
                preview = "\n".join(lines[:10])
                if len(lines) > 10:
                    preview += "\n... (output truncated)"
                console.print(preview)
                if explainer is not None:
                    console.print(str(explainer))

            return result, explainer
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]✗[/bold red] Error executing code: {e}")
            return None, None

    @staticmethod
    def _print_error_table(error_messages: list[str]) -> None:
        table = Table(title="Error Details", box=box.DOUBLE_EDGE)
        table.add_column("Attempt", style="cyan")
        table.add_column("Error", style="red")
        for i, msg in enumerate(error_messages, 1):
            table.add_row(str(i), msg)
        console.print(table)
