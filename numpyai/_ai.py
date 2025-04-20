import os
from rich.console import Console

c = Console()


class NumpyCodeGen:
    """
    Generates NumPy code from natural language using LLMs (Gemini, GPT, Claude).

    Parameters:
        provider_name (str):
            LLM provider name ("google", "openai", or "claude").
        model_name (Optional[str]):
             Specific model name to use (defaults per provider).
    """

    def __init__(self, provider_name="google", model_name=None) -> None:
        self._provider_name = provider_name.lower()

        if self._provider_name == "google":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Couldn't import `google-generativeai`.")

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self._model_name = model_name or "gemini-2.0-flash"
            self._system_prompt = (
                "You are a coding assistant who generates only NumPy and Python code."
            )
            self.messages = [{"role": "user", "parts": [self._system_prompt]}]

        elif self._provider_name == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("Couldn't import `openai`.")

            openai.api_key = os.getenv("OPENAI_API_KEY")
            self._model_name = model_name or "gpt-4-turbo"
            self.messages = [
                {
                    "role": "system",
                    "content": "You are a coding assistant who generates only NumPy and Python code.",
                }
            ]

        elif self._provider_name == "claude":
            try:
                import anthropic
            except ImportError:
                raise ImportError("Couldn't import `anthropic`.")

            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._model_name = model_name or "claude-3-opus-20240229"
            self._system_prompt = (
                "You are a coding assistant who generates only NumPy and Python code."
            )

        else:
            raise ValueError(f"Unknown provider: {self._provider_name}")

    def generate_response(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"

        if self._provider_name == "google":
            import google.generativeai as genai

            self.messages.append({"role": "user", "parts": [query]})
            model = genai.GenerativeModel(self._model_name)
            response = model.generate_content(self.messages)
            return getattr(response, "text", "Error: No response generated.")

        elif self._provider_name == "openai":
            import openai

            self.messages.append({"role": "user", "content": query})
            response = openai.ChatCompletion.create(
                model=self._model_name, messages=self.messages
            )
            return response["choices"][0]["message"]["content"].strip()

        elif self._provider_name == "claude":
            import anthropic

            response = self._client.messages.create(
                model=self._model_name,
                max_tokens=1024,
                system=self._system_prompt,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text.strip()

        return "Error: Unsupported provider."

    def generate_llm_prompt(self, query: str, metadata: dict) -> str:
        return f"""Generate NumPy code to perform the following operation: \n
        {query}. \n
        CRITICAL INSTRUCTIONS:
        1. The array is ALREADY defined as 'arr'. DO NOT create a new array with 'arr = ...'.
        2. DO NOT IMPORT any libraries except numpy (which is already imported).
        3. DO NOT use scipy, pandas, sklearn, or any other library. Use ONLY numpy functions.
        4. Return ONLY the code that operates on the existing 'arr' variable.
        5. There should always be exactly one variable named "output" which contains what 
        the user asked for.
        6. Beautifully Print what is important so the code is explainable and the users understand the output.
        7. Ensure data is properly cleaned before executing any code.
        8. As the very last option, if needed only, use sklearn (already imported as `sklearn`), matplotlib.pyplot (already imported as `plt`). 
        DO NOT USE IT IF NOT NECESSARY.
        
        The array has these properties:
        {metadata}
        
        CORRECT EXAMPLES:
        # Replace NaN values with zero
        arr[np.isnan(arr)] = 0
        
        # Calculate mean of array
        result = np.mean(arr)
        
        INCORRECT EXAMPLES (DO NOT DO THIS):
        # DON'T create a new array
        arr = np.array([1, 2, 3, 4, 5])
        
        # DON'T import scipy or other libraries
        from scipy import stats
        result = stats.zscore(arr)
        
        Your code must run using ONLY numpy functions. NO scipy, pandas, or other libraries.
        """

    def generate_llm_prompt_multiple(self, query: str, context: dict) -> str:
        """Generates an LLM prompt to handle multiple NumPy arrays in a session."""

        def format_metadata(metadata):
            """Convert metadata dictionary into a readable string format."""
            summary = [
                f"Shape: {metadata['shape']}, Dims: {metadata['dims']}, Type: {metadata['element_type']}",
                f"Size: {metadata['size']} elements, Memory: {metadata['byte_size']} bytes",
            ]

            if metadata.get("has_nan", False):
                summary.append("Contains NaN values")
            if metadata.get("has_inf", False):
                summary.append("Contains infinite values")
            if "min" in metadata and "max" in metadata:
                summary.append(f"Range: [{metadata['min']}, {metadata['max']}]")
            if "zeros_count" in metadata and "non_zeros_count" in metadata:
                summary.append(
                    f"Zero elements: {metadata['zeros_count']}, Non-zero elements: {metadata['non_zeros_count']}"
                )

            return "; ".join(summary)

        # Generate metadata descriptions
        array_descriptions = "\n".join(
            f"- **{name}**: {format_metadata(info['metadata'])}"
            for name, info in context.items()
        )

        return f"""Generate NumPy code to perform the following operation: \n
        {query}. \n
        CRITICAL INSTRUCTIONS:
        1. The following arrays are already defined: {', '.join(context.keys())}. DO NOT redefine them.
        2. DO NOT IMPORT any libraries except numpy (which is already imported).
        3. Use ONLY numpy functions. NO scipy, pandas, or any other library.
        4. The code should modify or compute results using the existing arrays.
        5. There must always be exactly one variable named "output" containing the result of the query.
        6. Ensure data is properly cleaned before executing any computation.
        7. Beautifully Print what is important so the code is explainable and the users understand the output.
        8. As the very last option, if needed only, use sklearn (already imported as `sklearn`), matplotlib.pyplot (already imported as `plt`). 
        DO NOT USE IT IF NOT NECESSARY.

        **Array Information:**
        {array_descriptions}

        CORRECT EXAMPLES:
        # Compute the mean of arr1
        output = np.mean(arr1)

        # Fill NaN values in arr2 with the mean of arr1
        arr2[np.isnan(arr2)] = np.mean(arr1)
        output = arr2

        INCORRECT EXAMPLES (DO NOT DO THIS):
        # DON'T create new arrays from scratch
        arr1 = np.array([1, 2, 3])  # WRONG!

        # DON'T import additional libraries
        from scipy.stats import zscore  # WRONG!
        output = zscore(arr1)

        Your code must be concise, clean, and use ONLY NumPy functions.
        """
