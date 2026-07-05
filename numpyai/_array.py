"""Single-array natural-language interface."""

from __future__ import annotations

import warnings
from operator import add, floordiv, matmul, mod, mul, pow, sub, truediv
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ._ai import DEFAULT_MODEL, CodeResponse, NumpyCodeGen
from ._exceptions import NumpyAIError
from ._utils import NumpyMetadataCollector, clean_code
from ._validator import NumpyValidator

console = Console()


class array:
    """A wrapper around ``numpy.ndarray`` with an AI-powered ``.chat()`` method.

    Parameters
    ----------
    data:
        The underlying NumPy array.
    verbose:
        If True, print every LLM step. If False, only print on the final attempt.
    model:
        Any model spec accepted by ``pydantic-ai`` (default:
        ``"google:gemini-2.5-flash"``).
    max_tries:
        Number of code-generation attempts before giving up (default: 3).
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        verbose: bool = False,
        model: Any = DEFAULT_MODEL,
        max_tries: int = 3,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(data).__name__}")

        self._data = data
        self._metadata_collector = NumpyMetadataCollector()
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen(model=model)
        self.MAX_TRIES = max_tries
        self.verbose = verbose

        self._output_metadata: dict = {}
        self.current_prompt: str | None = None
        self.metadata = self._metadata_collector.metadata(self._data)
        self._model = model

    # ------------------------------------------------------------------
    # numpy interop
    # ------------------------------------------------------------------
    def _apply_operator(self, other, op):
        other_data = other._data if isinstance(other, array) else other
        return array(op(self._data, other_data))

    def _apply_r_operator(self, other, op):
        return array(op(other, self._data))

    def __add__(self, other): return self._apply_operator(other, add)
    def __sub__(self, other): return self._apply_operator(other, sub)
    def __mul__(self, other): return self._apply_operator(other, mul)
    def __truediv__(self, other): return self._apply_operator(other, truediv)
    def __floordiv__(self, other): return self._apply_operator(other, floordiv)
    def __mod__(self, other): return self._apply_operator(other, mod)
    def __pow__(self, other): return self._apply_operator(other, pow)
    def __matmul__(self, other): return self._apply_operator(other, matmul)

    def __radd__(self, other): return self._apply_r_operator(other, add)
    def __rsub__(self, other): return self._apply_r_operator(other, sub)
    def __rmul__(self, other): return self._apply_r_operator(other, mul)
    def __rtruediv__(self, other): return self._apply_r_operator(other, truediv)
    def __rfloordiv__(self, other): return self._apply_r_operator(other, floordiv)
    def __rmod__(self, other): return self._apply_r_operator(other, mod)
    def __rpow__(self, other): return self._apply_r_operator(other, pow)
    def __rmatmul__(self, other): return self._apply_r_operator(other, matmul)

    def __getitem__(self, index):
        return array(self._data[index])

    def __setitem__(self, index, value) -> None:
        self._data[index] = value

    def __repr__(self) -> str:
        return f"numpyai.array(shape={self._data.shape}, dtype={self._data.dtype})"

    def __getattr__(self, name):
        attr = getattr(self._data, name)
        if callable(attr):
            def method_proxy(*args, **kwargs):
                result = attr(*args, **kwargs)
                return array(result) if isinstance(result, np.ndarray) else result
            return method_proxy
        return attr

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_array: np.ndarray) -> None:
        self._data = new_array
        self.metadata = self._metadata_collector.metadata(self._data)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def chat(self, query: str) -> Any:
        """Handle a natural-language query by generating and executing NumPy code."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue"))

        error_messages: list[str] = []
        prior_feedback: str | None = None
        for attempt in range(1, self.MAX_TRIES + 1):
            is_last = attempt == self.MAX_TRIES
            if self.verbose or is_last:
                console.print(f"[bold green]Attempt {attempt}/{self.MAX_TRIES}...[/bold green]")
            self.current_prompt = query

            try:
                response = self._generate_response(query, attempt, prior_feedback)
                if self.verbose or is_last:
                    console.print("[bold]Executing generated code...[/bold]")
                result, explainer = self._execute(response.code, self._data)

                if result is None:
                    error_messages.append(f"Try {attempt}: Code execution returned None")
                    prior_feedback = "code execution produced no `output` variable"
                    if self.verbose or is_last:
                        console.print(f"[bold red]✗[/bold red] Attempt {attempt} failed: execution returned None")
                    continue

                self._output_metadata = self._metadata_collector.collect_output_metadata(result)

                judgment = self._code_generator.judge(
                    query=query, code=response.code, metadata=str(explainer or "")
                )
                if self.verbose or is_last:
                    self._print_judgment(judgment)

                if judgment.interprets_query_correctly:
                    console.print("[bold green]✓[/bold green] Judgment passed!")
                    return result

                prior_feedback = f"judgment rejected: {judgment.reason}"
                error_messages.append(f"Try {attempt}: {prior_feedback}")

            except Exception as e:
                error_messages.append(f"Try {attempt}: {e}")
                prior_feedback = f"exception in previous attempt: {e}"
                if self.verbose or is_last:
                    console.print(f"[bold red]✗[/bold red] Attempt {attempt} failed: {e}")

        self._print_error_table(error_messages)
        warnings.warn(
            f"Validation failed after {self.MAX_TRIES} attempts. Please check the validity of the code.",
            stacklevel=2,
        )
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _print_error_table(error_messages: list[str]) -> None:
        table = Table(title="Error Details", box=box.DOUBLE_EDGE)
        table.add_column("Attempt", style="cyan")
        table.add_column("Error", style="red")
        for i, msg in enumerate(error_messages, 1):
            table.add_row(str(i), msg)
        console.print(table)

    @staticmethod
    def _print_judgment(j) -> None:
        mark = "[green]✓[/green]" if j.interprets_query_correctly else "[red]✗[/red]"
        body = j.reason or "correctly interprets the query"
        console.print(Panel(f"{mark} {body}", title="Judgment", border_style="magenta"))

    def _generate_response(
        self,
        query: str,
        attempt: int,
        prior_feedback: str | None = None,
    ) -> CodeResponse:
        prompt = self._code_generator.prompt_single(
            query=query, metadata=self.metadata, prior_feedback=prior_feedback
        )
        response = self._code_generator.generate_code(prompt)
        response = CodeResponse(
            code=clean_code(response.code),
            explanation=response.explanation,
        )

        if self.verbose or attempt == self.MAX_TRIES:
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

    def _execute(self, code: str, args) -> tuple[Any, Any]:
        """Execute generated code in a controlled namespace.

        Returns ``(result, explainer)``. On error, returns ``(None, None)``.
        """
        try:
            local_vars: dict[str, Any] = {"np": np}
            try:
                import matplotlib.pyplot as plt  # noqa: F401
                local_vars["plt"] = plt
            except ImportError:
                pass
            try:
                import sklearn  # noqa: F401
                local_vars["sklearn"] = sklearn
            except ImportError:
                pass

            if isinstance(args, dict):
                local_vars.update(args)
            else:
                local_vars["arr"] = args

            exec(code, {"__builtins__": __builtins__}, local_vars)
            result = local_vars.get("output")
            explainer = local_vars.get("metadata")

            if self.verbose:
                if result is not None:
                    console.print("\n".join(str(result).split("\n")[:10]))
                if explainer is not None:
                    console.print(str(explainer))

            return result, explainer

        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]✗[/bold red] Error executing code: {e}")
            return None, None
