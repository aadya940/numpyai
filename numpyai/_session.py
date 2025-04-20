from typing import List, Dict, Union, Any
import numpy as np
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from ._array import array
from ._utils import NumpyMetadataCollector, clean_code
from ._ai import NumpyCodeGen
from ._exceptions import NumpyAIError
from ._validator import NumpyValidator

import sklearn
import matplotlib.pyplot as plt

# Initialize rich console
console = Console()

console.print(
    "[yellow]Ensure your API KEY is set for your LLM as an environment variable.[/yellow]"
)


class NumpyAISession:
    """Session to handle chatting with multiple arrays.

    Args:
        data: List[numpy.ndarray]
            The data of the array class.
        verbose: boolean
            Shows all code executions if True, else only shows the final execution.
        provider_name (str):
            LLM provider name ("google", "openai", or "claude").
        model_name (Optional[str]):
             Specific model name to use (defaults per provider).
    """

    MAX_TRIES = 3  # Class constant instead of instance attribute

    def __init__(
        self,
        data: List[Union[np.ndarray, array]],
        verbose=False,
        provider_name="google",
        model_name=None,
    ) -> None:
        self._context: Dict[str, Dict[str, Union[np.ndarray, Dict]]] = {}
        self._metadata_collector = NumpyMetadataCollector()
        self._code_generator = NumpyCodeGen()
        self._validator = NumpyValidator()

        self._initialize_arrays(data)
        self.current_prompt = None
        self._output_metadata = {}
        self.verbose = verbose

        self._cur_provider = provider_name
        self._cur_model = model_name

    def _initialize_arrays(self, data: List[Union[np.ndarray, array]]) -> None:
        """Stores input arrays with default names arr1, arr2, etc."""
        for i, arr in enumerate(data, start=1):
            if isinstance(arr, array):
                arr = arr.get_array()

            self._context[f"arr{i}"] = {
                "array": arr,
                "metadata": self._metadata_collector.metadata(arr),
            }

    def validate_output(self, query: str, output: Any, error=None) -> bool:
        """Validates the output of a NumPy operation."""
        input_metadata = {
            name: info["metadata"] for name, info in self._context.items()
        }
        output_metadata = self._metadata_collector.collect_output_metadata(output)

        validation_prompt = self._validator.generate_validation_prompt_multiple(
            query, input_metadata, output_metadata, error=error
        )
        validation_code = self._code_generator.generate_response(validation_prompt)
        validation_code = self._clean_code(validation_code)

        # Wrap the validation code in Syntax for better display
        syntax = Syntax(validation_code, "python", theme="monokai", line_numbers=True)
        if self.verbose:
            console.print(Panel(syntax, title="Validation Code", border_style="blue"))

        # Execute the validation code
        validation_result = self.execute_numpy_code(validation_code, code_out=output)

        if self.verbose:
            console.print(
                Panel(
                    (
                        "Validation Successful."
                        if validation_result
                        else "Validation Failed."
                    ),
                    style="bold green" if validation_result else "bold red",
                )
            )

        return validation_result

    def _clean_code(self, code: str) -> str:
        """Remove code blocks and extra whitespace from LLM response."""
        return clean_code(code)

    def generate_numpy_code(self, query: str, context: Dict) -> str:
        """Generate valid NumPy code from the query."""
        prompt = self._code_generator.generate_llm_prompt_multiple(
            query=query, context=context
        )
        llm_response = self._code_generator.generate_response(prompt)

        return self.assert_is_code(llm_response)

    def chat(self, query: str) -> Any:
        """Handles user queries by generating and executing NumPy code with retries."""

        console.print(
            Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue")
        )

        exceptions = []
        error_messages = []

        for attempt in range(1, self.MAX_TRIES + 1):
            if self.verbose:
                console.print(
                    Panel(
                        f"[bold green]Attempt {attempt}/{self.MAX_TRIES}...[/bold green]",
                        border_style="yellow",
                    )
                )

            try:
                self.current_prompt = query
                code = self.generate_numpy_code(query, self._context)

                result = self.execute_numpy_code(code)

                if result is None:
                    error_messages.append(
                        f"Attempt {attempt}: Execution returned None."
                    )
                    if self.verbose:
                        console.print(
                            Panel(
                                f"[bold red]✗[/bold red] Execution returned None.",
                                border_style="red",
                            )
                        )

                    continue

                if self.validate_output(query, result, error=error_messages[-1]):
                    if self.verbose or (attempt == self.MAX_TRIES):
                        console.print(
                            Panel(
                                f"[bold green]Output\n {result if not isinstance(result, np.ndarray) else type(result)}",
                                border_style="yellow",
                            )
                        )
                    return result

                raise NumpyAIError("Validation failed for the output.")

            except Exception as e:
                exceptions.append(str(e))
                error_messages.append(f"Attempt {attempt}: {str(e)}")
                if self.verbose or (attempt == self.MAX_TRIES):
                    console.print(
                        Panel(
                            f"[bold red]✗[/bold red] Attempt {attempt} failed: {str(e)}",
                            border_style="red",
                        )
                    )

        # Print error details before raising the final exception
        self._print_error_table(error_messages)
        raise NumpyAIError(f"Validation failed after {self.MAX_TRIES} attempts.")

    def assert_is_code(self, llm_response: str) -> str:
        """Ensure LLM response is valid Python/NumPy code."""
        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            code = self._clean_code(llm_response)
            try:
                if self._validator.validate_code(code):
                    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                    if self.verbose:
                        console.print(
                            Panel(
                                syntax,
                                title="[bold]Generated Code[/bold]",
                                border_style="blue",
                            )
                        )

                    return code

            except SyntaxError as e:
                error_messages.append(f"Syntax error: {str(e)}")
                tries += 1
                llm_response = self._code_generator.generate_response(
                    self.current_prompt
                )
                continue

            tries += 1

        raise NumpyAIError(
            f"Error generating valid code after {self.MAX_TRIES} attempts."
        )

    def _print_error_table(self, error_messages: List[str]) -> None:
        """Displays an error summary table."""
        error_table = Table(title="Error Details", box=box.DOUBLE_EDGE)
        error_table.add_column("Attempt", style="cyan")
        error_table.add_column("Error", style="red")

        for i, msg in enumerate(error_messages, 1):
            error_table.add_row(str(i), msg)

        console.print(error_table)

    def execute_numpy_code(self, code: str, code_out=None) -> Any:
        """Executes generated NumPy code within the session's context."""
        local_vars = {"np": np}
        for name, info in self._context.items():
            local_vars[name] = info["array"]  # Add arrays as variables

        if code_out is not None:
            local_vars["code_out"] = code_out

        try:
            exec_globals = {
                "__builtins__": __builtins__,
                "np": np,
                "plt": plt,
                "sklearn": sklearn,
            }
            exec(code, exec_globals, local_vars)  # Execute code in controlled scope
            result = local_vars.get("output", None)

            if result is not None:
                # Show only first 10 lines of output
                output_lines = str(result).split("\n")
                preview = "\n".join(output_lines[:10])
                if len(output_lines) > 10:
                    preview += "\n... (output truncated)"

                if self.verbose:
                    console.print(preview)
                return result

            return None

        except Exception as e:
            raise RuntimeError(f"Error executing generated code: {e}")
