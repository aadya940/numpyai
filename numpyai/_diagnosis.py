from ._array import array
from ._session import NumpyAISession
from ._ai import NumpyCodeGen
from ._exceptions import NumpyAIError
from ._utils import clean_code

from typing import Union, Optional, Any
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import ast

console = Console()


class Diagnosis:
    """Class for data analysis steps for the given NumpyAI objects."""

    def __init__(self, data: Union[array, NumpyAISession]):
        if isinstance(data, array):
            self._type = "single"
            self._metadata: Any = data.metadata
        elif isinstance(data, NumpyAISession):
            self._type = "multi"
            self._metadata: Any = data._context
        else:
            raise ValueError("`data` must be a NumpyAIArray or NumpyAISession.")

        self.MAX_TRIES = 3
        self._cur_tries = 0

        self._code_generator = NumpyCodeGen()

    def _diagnosis_prompt(self, objective: Optional[str] = None) -> str:
        data_type = (
            "NumPy array" if self._type == "single" else "collection of NumPy arrays"
        )

        prompt = f"""
        # NumPy Data Analysis Assistant
        
        You are analyzing {data_type} with the following metadata:
        ```
        {self._metadata}
        ```
        
        Based on this data, provide a clear analytical strategy using NumPy operations.
        """

        if objective:
            prompt += f"""
        
        SPECIFIC TASK: {objective}
        """

        prompt += """
        
        ## Response Guidelines:
        1. Provide a numbered list of specific steps in plain English
        2. Focus on analytical insights rather than code implementation
        3. When suggesting NumPy operations, name the relevant functions without showing syntax
        4. Include reasoning for each recommended approach
        5. Be specific about techniques for handling any data issues (missing values, outliers, etc.)
        6. If machine learning is mentioned, suggest appropriate models and preprocessing steps
        7. Be concise yet thorough - each step should be actionable
        8. Specify what to do after executing each step. For instance, if you tell me to compute correlation,
        also tell us how the output result can be used in context with the specific task.
        9. Specify how to use the result of each step for diagnosis. For instance, if we calculate outliers in
        a specific step. How to fix the outliers if they are present.
        10. Your response should be a valid python list of strings only.
        
        Do not include introductions, conclusions, or code examples. Start directly with the numbered steps.
        """

        return prompt.strip()

    def steps(self, task: Optional[str] = None) -> str:
        """Return thoughtful and exact data analysis steps for the given data."""

        prompt = self._diagnosis_prompt(objective=task)
        self._cur_tries = 0

        while self._cur_tries < self.MAX_TRIES:
            response = self._code_generator.generate_response(prompt)
            response = clean_code(response)

            console.rule("[bold blue]LLM Response[/bold blue]")
            console.print(
                Panel.fit(
                    Markdown(response), title="Data Analysis Steps", border_style="cyan"
                )
            )

            # Try to parse the response as a Python list
            try:
                # Use ast.literal_eval to safely convert string to list
                parsed_response = ast.literal_eval(response)
                if isinstance(parsed_response, list):
                    return parsed_response
            except (ValueError, SyntaxError):
                pass  # Continue to next attempt if parsing fails

            self._cur_tries += 1

        raise NumpyAIError(
            f"[bold red]Failed after {self.MAX_TRIES} attempts.[/bold red]"
        )
