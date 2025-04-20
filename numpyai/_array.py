# Import necessary libraries
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box
import numpy as np
from operator import add, sub, mul, truediv, floordiv, mod, pow, matmul
import matplotlib.pyplot as plt
import sklearn

import re
from typing import Any, Dict, List, Optional, Union, Tuple

from ._validator import NumpyValidator
from ._ai import NumpyCodeGen
from ._utils import NumpyMetadataCollector, clean_code
from ._exceptions import NumpyAIError

# Initialize rich console
console = Console()

console.print(
    "[yellow]Ensure your API KEY is set for your LLM as an environment variable.[/yellow]"
)


class array:
    """A wrapper around `numpy.ndarray` providing AI-powered functionalities
    and extended operations.

    Args:
        data: numpy.ndarray
            The data of the array class.
        verbose: bool, default=True
            If False, only show rich console outputs when the try is the last try,
            successful, or fails in the last try.
        provider_name (str):
            LLM provider name ("google", "openai", or "claude").
        model_name (Optional[str]):
             Specific model name to use (defaults per provider).
    """

    def __init__(self, data, verbose=False, provider_name="google", model_name=None):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._metadata_collector = NumpyMetadataCollector()
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen(
            provider_name=provider_name, model_name=model_name
        )
        self.MAX_TRIES = 3
        self.verbose = verbose

        self._output_metadata = {}

        self.current_prompt = None
        self.metadata = self._metadata_collector.metadata(self._data)

        self._cur_provider = provider_name
        self._cur_model = model_name

    def _apply_operator(self, other, op):
        other_data = other._data if isinstance(other, array) else other
        return array(op(self._data, other_data))

    def _apply_r_operator(self, other, op):
        return array(op(other, self._data))

    def __add__(self, other):
        return self._apply_operator(other, add)

    def __sub__(self, other):
        return self._apply_operator(other, sub)

    def __mul__(self, other):
        return self._apply_operator(other, mul)

    def __truediv__(self, other):
        return self._apply_operator(other, truediv)

    def __floordiv__(self, other):
        return self._apply_operator(other, floordiv)

    def __mod__(self, other):
        return self._apply_operator(other, mod)

    def __pow__(self, other):
        return self._apply_operator(other, pow)

    def __matmul__(self, other):
        return self._apply_operator(other, matmul)

    def __radd__(self, other):
        return self._apply_r_operator(other, add)

    def __rsub__(self, other):
        return self._apply_r_operator(other, sub)

    def __rmul__(self, other):
        return self._apply_r_operator(other, mul)

    def __rtruediv__(self, other):
        return self._apply_r_operator(other, truediv)

    def __rfloordiv__(self, other):
        return self._apply_r_operator(other, floordiv)

    def __rmod__(self, other):
        return self._apply_r_operator(other, mod)

    def __rpow__(self, other):
        return self._apply_r_operator(other, pow)

    def __rmatmul__(self, other):
        return self._apply_r_operator(other, matmul)

    def __getitem__(self, index):
        result = self._data[index]
        return array(result)

    def __setitem__(self, index, value):
        self._data[index] = value

    def __repr__(self):
        return f"numpyai.array(shape={self._data.shape}, dtype={self._data.dtype})"

    def __getattr__(self, name):
        """Implements numpy methods using method forwarding."""
        attr = getattr(self._data, name)
        if callable(attr):
            # Wrap it in a function to allow forwarding kwargs
            def method_proxy(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    return array(result)
                return result

            return method_proxy
        return attr  # Return as-is if it's not callable

    # Utility methods
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_array):
        self._data = new_array
        self.metadata = self._metadata_collector.metadata(self._data)

    def chat(self, query):
        """Handles user queries by generating and executing NumPy code."""
        assert isinstance(query, str)
        console.print(
            Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue")
        )

        error_messages = []
        for attempt in range(1, self.MAX_TRIES + 1):
            # Only display attempt messages if verbose or last attempt
            if self.verbose or attempt == self.MAX_TRIES:
                console.print(
                    f"[bold green]Attempt {attempt}/{self.MAX_TRIES}...[/bold green]"
                )
            self.current_prompt = query

            try:
                _code = self.generate_numpy_code(query, attempt)
                if not isinstance(_code, str):
                    continue

                if self.verbose or attempt == self.MAX_TRIES:
                    console.print("[bold]Executing generated code...[/bold]")
                _res = self.execute_numpy_code(_code, self._data)
                if _res is None:
                    error_messages.append(
                        f"Try {attempt}: Code execution returned None"
                    )
                    if self.verbose or attempt == self.MAX_TRIES:
                        console.print(
                            f"[bold red]✗[/bold red] Attempt {attempt} failed: Code execution returned None"
                        )
                    continue

                self._output_metadata = (
                    self._metadata_collector.collect_output_metadata(_res)
                )
                _test_code = clean_code(
                    self._code_generator.generate_response(
                        self._validator.generate_validation_prompt(
                            query, self.metadata, self._output_metadata
                        )
                    )
                )

                if self.verbose or attempt == self.MAX_TRIES:
                    console.print(
                        Panel(
                            Syntax(
                                _test_code, "python", theme="monokai", line_numbers=True
                            ),
                            title="[bold]Validation Code[/bold]",
                            border_style="green",
                        )
                    )

                _test_response = self.execute_numpy_code(
                    _test_code, {"arr": self._data, "code_out": _res}
                )
                if self._is_valid_test_response(_test_response):
                    console.print("[bold green]✓[/bold green] Validation successful!")
                    return _res

            except Exception as e:
                error_messages.append(f"Try {attempt}: {str(e)}")
                if self.verbose or attempt == self.MAX_TRIES:
                    console.print(
                        f"[bold red]✗[/bold red] Attempt {attempt} failed: {str(e)}"
                    )

        self._print_error_table(error_messages)
        raise NumpyAIError(
            f"[bold red]Failed after {self.MAX_TRIES} attempts.[/bold red]"
        )

    def _is_valid_test_response(self, response):
        """Validates the test response based on type and boolean evaluation."""
        if isinstance(response, bool):
            return response
        if isinstance(response, np.ndarray):
            return response.size == 1 and bool(response.item()) or response.all()
        try:
            return bool(response)
        except (ValueError, TypeError):
            return False

    def _print_error_table(self, error_messages):
        """Displays an error summary table."""
        error_table = Table(title="Error Details", box=box.DOUBLE_EDGE)
        error_table.add_column("Attempt", style="cyan")
        error_table.add_column("Error", style="red")

        for i, msg in enumerate(error_messages, 1):
            error_table.add_row(str(i), msg)

        console.print(error_table)

    def generate_numpy_code(self, query, attempt=1):
        """Generate valid NumPy code from the query."""
        pr = self._code_generator.generate_llm_prompt(
            query=query, metadata=self.metadata
        )
        llm_res = self._code_generator.generate_response(pr)

        # Only display the code panel if verbose or last attempt
        if self.verbose or attempt == self.MAX_TRIES:
            syntax = Syntax(llm_res, "python", theme="monokai", line_numbers=True)
            console.print(
                Panel(syntax, title="[bold]Generated Code[/bold]", border_style="blue")
            )

        return self.assert_is_code(llm_res, attempt)

    def assert_is_code(self, llm_response, attempt=1):
        """Ensure LLM response is valid Python/NumPy code."""
        if not isinstance(llm_response, str):
            if self.verbose or attempt == self.MAX_TRIES:
                console.print("[bold red]✗[/bold red] LLM response is not a string")
            raise ValueError("LLM response is not a string")

        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            code = clean_code(r"```(\w+)?", "", llm_response)
            try:
                if self._validator.validate_code(code):
                    return code
            except SyntaxError as e:
                error_messages.append(f"Syntax error: {str(e)}")
                tries += 1
                if self.verbose or tries == self.MAX_TRIES:
                    console.print(f"[bold red]✗[/bold red] Syntax error: {str(e)}")
                    console.print("[yellow]Regenerating code...[/yellow]")
                llm_response = self._code_generator.generate_response(
                    self.current_prompt
                )
                continue

            tries += 1

        # Always show the error table at the end of all attempts
        error_table = Table(title="Code Validation Errors", box=box.SIMPLE)
        error_table.add_column("Attempt", style="cyan")
        error_table.add_column("Error", style="red")

        for i, msg in enumerate(error_messages):
            error_table.add_row(f"{i+1}", msg)

        console.print(error_table)

        raise NumpyAIError(
            f"[bold red]Error generating valid code after {self.MAX_TRIES} attempts.[/bold red]"
        )

    def execute_numpy_code(self, code, args):
        """Execute the generated code safely.

        Args:
            code: The code to execute
            args: Either the array itself or a dict containing variables for execution
        """
        try:
            local_vars = {"np": np, "plt": plt, "sklearn": sklearn}

            if isinstance(args, dict):
                local_vars.update(args)
            else:
                local_vars["arr"] = args

            # Execute the code block
            exec(code, {"__builtins__": __builtins__}, local_vars)
            result = local_vars.get("output")

            if result is not None and self.verbose:
                console.print("\n".join(str(result).split("\n")[:10]))
            return result

        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]✗[/bold red] Error executing code: {str(e)}")
            return None
