from ._array import array
from ._session import NumpyAISession
from ._ai import NumpyCodeGen

from typing import Union, Optional, Any
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class Diagnosis:
    def __init__(self, data: Union[array, NumpyAISession]):
        if isinstance(data, array):
            self._type = "single"
            self._metadata: Any = data.metadata
        elif isinstance(data, NumpyAISession):
            self._type = "multi"
            self._metadata: Any = data._context
        else:
            raise ValueError("`data` must be a NumpyAIArray or NumpyAISession.")

        self._code_generator = NumpyCodeGen()

    def _diagnosis_prompt(self, objective: Optional[str] = None) -> str:
        prompt = f"""
        You are working with a {'NumPy array' if self._type == 'single' else 'list of NumPy arrays'}.
        Based on the metadata provided below, outline a complete, step-by-step plan for analyzing the data using only NumPy.

        Metadata:
        {self._metadata}

        Guidelines:
        1. Provide precise, actionable and very specific steps.
        2. Avoid including anything beyond the requested analysis.
        3. Include brief explanations where helpful.
        4. Use only NumPy for all operations. No other library should be referenced.
        5. Respond in plain English—no code.
        6. Ensure every step can be implemented using the NumPy library only.
        7. Only return the points, nothing else.

        You should be diagnostic in nature. For example, if you're guiding the user to impute the NaN values, you should
        also tell him which strategy to use and why.
        """.strip()

        if objective:
            prompt += f"\n\nObjective:\n{objective}"

        return prompt

    def steps(self, task: Optional[str] = None) -> str:
        """Return thoughtful and exact data analysis steps for the given data."""

        prompt = self._diagnosis_prompt(objective=task)
        response = self._code_generator.generate_response(prompt)

        console.rule("[bold blue]LLM Response[/bold blue]")
        console.print(
            Panel.fit(
                Markdown(response), title="Data Analysis Steps", border_style="cyan"
            )
        )

        return response
