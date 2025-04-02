import google.generativeai as genai
import os
from rich.console import Console

c = Console()


class NumpyCodeGen:
    """Generates Numpy code for execution."""

    def __init__(self, model_name=None) -> None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        if not model_name:
            self._model_name = "gemini-2.0-flash"  # Use a valid Gemini model
        else:
            self._model_name = model_name

        self._system_prompt = (
            "You are a coding assistant who generates only NumPy and Python code."
        )
        self.messages = [{"role": "user", "parts": [self._system_prompt]}]

    def generate_response(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"
        self.messages.append({"role": "user", "parts": [query]})

        model = genai.GenerativeModel(self._model_name)  # Initialize the model
        response = model.generate_content(self.messages)  # Generate response

        if not response or not hasattr(response, "text"):
            return "Error: No response generated."

        return response.text  # Gemini responses have a `.text` attribute
