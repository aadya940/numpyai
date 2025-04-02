# Import necessary libraries
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box
import numpy as np
import re
from typing import Any, Dict, List, Optional, Union, Tuple

from ._validator import NumpyValidator
from ._ai import NumpyCodeGen
from ._utils import NumpyMetadataCollector
from ._exceptions import NumpyAIError

# Initialize rich console
console = Console()

console.print(
    "[yellow]Ensure your API KEY is set for your LLM as an environment variable.[/yellow]"
)


class array:
    """A wrapper around `numpy.ndarray` providing AI-powered functionalities
    and extended operations.
    """

    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._metadata_collector = NumpyMetadataCollector()
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen()
        self.MAX_TRIES = 3

        self._output_metadata = {}

        self.current_prompt = None
        self.metadata = self._metadata_collector.metadata(self._data)

    # Basic operators with scalar support
    def __add__(self, other):
        if isinstance(other, array):
            result = self._data + other._data
        else:
            result = self._data + other
        return array(result)

    def __radd__(self, other):
        result = other + self._data
        return array(result)

    def __sub__(self, other):
        if isinstance(other, array):
            result = self._data - other._data
        else:
            result = self._data - other
        return array(result)

    def __rsub__(self, other):
        result = other - self._data
        return array(result)

    def __mul__(self, other):
        if isinstance(other, array):
            result = self._data * other._data
        else:
            result = self._data * other
        return array(result)

    def __rmul__(self, other):
        result = other * self._data
        return array(result)

    def __truediv__(self, other):
        if isinstance(other, array):
            result = self._data / other._data
        else:
            result = self._data / other
        return array(result)

    def __rtruediv__(self, other):
        result = other / self._data
        return array(result)

    def __floordiv__(self, other):
        if isinstance(other, array):
            result = self._data // other._data
        else:
            result = self._data // other
        return array(result)

    def __rfloordiv__(self, other):
        result = other // self._data
        return array(result)

    def __pow__(self, other):
        if isinstance(other, array):
            result = self._data**other._data
        else:
            result = self._data**other
        return array(result)

    def __rpow__(self, other):
        result = other**self._data
        return array(result)

    def __mod__(self, other):
        if isinstance(other, array):
            result = self._data % other._data
        else:
            result = self._data % other
        return array(result)

    def __rmod__(self, other):
        result = other % self._data
        return array(result)

    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        if isinstance(other, array):
            result = self._data @ other._data
        else:
            result = self._data @ other
        return array(result)

    def __rmatmul__(self, other):
        result = other @ self._data
        return array(result)

    def __neg__(self):
        result = -self._data
        return array(result)

    def __pos__(self):
        result = +self._data
        return array(result)

    def __abs__(self):
        result = abs(self._data)
        return array(result)

    def __getitem__(self, index):
        result = self._data[index]
        return array(result)

    def __setitem__(self, index, value):
        self._data[index] = value

    def __repr__(self):
        return f"numpyai.array({repr(self._data.shape)})"

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
    def get_array(self):
        """Returns the underlying Numpy Array of the `numpyai.array` class."""
        return self._data

    def set_array(self, new_array):
        """Sets the underlying Numpy Array to `new_array`."""
        self._data = new_array
        self.metadata = self._metadata_collector.metadata(self._data)

    def chat(self, query):
        """Handles user queries by generating and executing NumPy code."""
        assert isinstance(query, str)

        console.print(
            Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue")
        )

        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            console.print(
                f"[bold green]Attempt {tries+1}/{self.MAX_TRIES}...[/bold green]"
            )
            self.current_prompt = query
            try:
                _code = self.generate_numpy_code(query)
                if isinstance(_code, str):
                    console.print("[bold]Executing generated code...[/bold]")
                    _res = self.execute_numpy_code(_code, self._data)
                    if _res is None:
                        error_messages.append(
                            f"Try {tries+1}: Code execution returned None"
                        )
                        console.print(
                            f"[bold red]✗[/bold red] Attempt {tries+1} failed: Code execution returned None"
                        )
                        tries += 1
                        continue

                    self._output_metadata = (
                        self._metadata_collector.collect_output_metadata(_res)
                    )
                    # Generate and run test code
                    _testing_prompt = self._validator.generate_validation_prompt(
                        query=query,
                        metadata=self.metadata,
                        output_metadata=self._output_metadata,
                    )
                    _testing_code = self._code_generator.generate_response(
                        _testing_prompt
                    )
                    _testing_code = re.sub(r"```(\w+)?", "", _testing_code).strip()

                    console.print(
                        Panel(
                            Syntax(
                                _testing_code,
                                "python",
                                theme="monokai",
                                line_numbers=True,
                            ),
                            title="[bold]Validation Code[/bold]",
                            border_style="green",
                        )
                    )

                    _test_args = {"arr": self._data, "code_out": _res}
                    _test_response = self.execute_numpy_code(_testing_code, _test_args)

                    # Fix the boolean check with proper handling for arrays
                    if _test_response is not None:
                        if isinstance(_test_response, bool):
                            if _test_response:
                                console.print(
                                    "[bold green]✓[/bold green] Validation successful!"
                                )
                                if isinstance(_res, np.ndarray):
                                    return _res
                                return _res
                        elif isinstance(_test_response, np.ndarray):
                            # Handle array truth value - use all() or any() based on your validation needs
                            if _test_response.size == 1:
                                if bool(_test_response.item()):
                                    console.print(
                                        "[bold green]✓[/bold green] Validation successful!"
                                    )
                                    if isinstance(_res, np.ndarray):
                                        return _res
                                    return _res
                            elif (
                                _test_response.all()
                            ):  # or .any() depending on validation requirements
                                console.print(
                                    "[bold green]✓[/bold green] Validation successful!"
                                )
                                if isinstance(_res, np.ndarray):
                                    return _res
                                return _res
                        else:
                            # For other non-None return types, evaluate as boolean
                            try:
                                if bool(_test_response):
                                    console.print(
                                        "[bold green]✓[/bold green] Validation successful!"
                                    )
                                    if isinstance(_res, np.ndarray):
                                        return _res
                                    return _res
                            except (ValueError, TypeError):
                                # If boolean conversion fails, consider it a failed test
                                error_messages.append(
                                    f"Try {tries+1}: Validation failed - couldn't convert test response to boolean"
                                )
                                console.print(
                                    f"[bold red]✗[/bold red] Validation failed - couldn't convert test response to boolean"
                                )
            except Exception as e:
                error_messages.append(f"Try {tries+1}: {str(e)}")
                console.print(
                    f"[bold red]✗[/bold red] Attempt {tries+1} failed: {str(e)}"
                )

            tries += 1

        # More detailed error message with history of what went wrong
        error_table = Table(title="Error Details", box=box.DOUBLE_EDGE)
        error_table.add_column("Attempt", style="cyan")
        error_table.add_column("Error", style="red")

        for i, msg in enumerate(error_messages):
            error_table.add_row(f"{i+1}", msg)

        console.print(error_table)

        raise NumpyAIError(
            f"[bold red]Failed to generate correct response after {self.MAX_TRIES} attempts.[/bold red]"
        )

    def generate_numpy_code(self, query):
        """Generate valid NumPy code from the query."""
        pr = self._code_generator.generate_llm_prompt(
            query=query, metadata=self.metadata
        )
        llm_res = self._code_generator.generate_response(pr)

        syntax = Syntax(llm_res, "python", theme="monokai", line_numbers=True)
        console.print(
            Panel(syntax, title="[bold]Generated Code[/bold]", border_style="blue")
        )

        return self.assert_is_code(llm_res)

    def assert_is_code(self, llm_response):
        """Ensure LLM response is valid Python/NumPy code."""
        if not isinstance(llm_response, str):
            console.print("[bold red]✗[/bold red] LLM response is not a string")
            raise ValueError("LLM response is not a string")

        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            code = re.sub(r"```(\w+)?", "", llm_response).strip()
            try:
                if self._validator.validate_code(code):
                    return code
            except SyntaxError as e:
                error_messages.append(f"Syntax error: {str(e)}")
                tries += 1
                console.print(f"[bold red]✗[/bold red] Syntax error: {str(e)}")
                console.print("[yellow]Regenerating code...[/yellow]")
                llm_response = self._code_generator.generate_response(
                    self.current_prompt
                )
                continue

            tries += 1

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
            local_vars = {"np": np}

            if isinstance(args, dict):
                local_vars.update(args)
            else:
                local_vars["arr"] = args

            # Execute the code block
            exec(code, {"__builtins__": __builtins__}, local_vars)
            result = local_vars.get("output")

            if result is not None:
                console.print("\n".join(str(result).split("\n")[:10]))
            return result

        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Error executing code: {str(e)}")
            return None
