"""Data-analysis step generator for numpyai objects."""

from __future__ import annotations

import ast
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ._ai import NumpyCodeGen
from ._array import array
from ._exceptions import NumpyAIError
from ._session import NumpyAISession
from ._utils import clean_code

console = Console()


class Diagnosis:
    """Suggest data-analysis steps for a :class:`array` or :class:`NumpyAISession`."""

    def __init__(self, data: array | NumpyAISession, *, max_tries: int = 3) -> None:
        if isinstance(data, array):
            self._type = "single"
            self._metadata: Any = data.metadata
        elif isinstance(data, NumpyAISession):
            self._type = "multi"
            self._metadata = data._context
        else:
            raise TypeError("`data` must be a numpyai.array or NumpyAISession")

        self.MAX_TRIES = max_tries
        self._code_generator = NumpyCodeGen(model=data._model)

    def _diagnosis_prompt(self, objective: str | None = None) -> str:
        data_type = "NumPy array" if self._type == "single" else "collection of NumPy arrays"

        prompt = f"""# NumPy Data Analysis Assistant

You are analyzing a {data_type} with the following metadata:
```
{self._metadata}
```

Based on this data, provide a clear analytical strategy using NumPy operations.
"""

        if objective:
            prompt += f"\nSPECIFIC TASK: {objective}\n"

        prompt += """
## Response Guidelines
1. Provide a numbered list of specific steps in plain English.
2. Focus on analytical insights rather than code implementation.
3. Name relevant NumPy functions without showing syntax.
4. Include reasoning for each recommended approach.
5. Be specific about handling missing values, outliers, etc.
6. If machine learning is mentioned, suggest models and preprocessing steps.
7. Be concise yet thorough - each step should be actionable.
8. Specify what to do after each step and how its output feeds the next.
9. Say how to use the result of each step for diagnosis.
10. Your response MUST be a valid Python list of strings - nothing else.

Do not include introductions, conclusions, or code examples.
"""

        return prompt.strip()

    def steps(self, task: str | None = None) -> list[str]:
        """Return a list of data-analysis steps for the given data."""
        prompt = self._diagnosis_prompt(objective=task)

        for _ in range(self.MAX_TRIES):
            response = clean_code(self._code_generator.generate_text(prompt))

            console.rule("[bold blue]LLM Response[/bold blue]")
            console.print(
                Panel.fit(
                    Markdown(response),
                    title="Data Analysis Steps",
                    border_style="cyan",
                )
            )

            try:
                parsed = ast.literal_eval(response)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                continue

        raise NumpyAIError(f"Failed after {self.MAX_TRIES} attempts.")
