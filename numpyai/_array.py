import numpy as np
import re
from rich.console import Console

from ._validator import NumpyValidator
from ._ai import NumpyCodeGen
from ._utils import collect_metadata, collect_output_metadata
from ._prompts import numpy_single_array_llm_prompt, validate_llm_output

c = Console()


class array:
    def __init__(self, data, **kwargs):
        """NumpyAI wrapper for `numpy.ndarray`."""
        if not isinstance(data, np.ndarray):
            self._data = np.array(data, **kwargs)
        else:
            self._data = data

        self.metadata = self._collect_metadata(self._data)
        self.current_prompt = None
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen()
        self.MAX_TRIES = 3

        self._output_metadata = {}

    def _collect_metadata(self, data):
        """Collect comprehensive metadata about the NumPy array."""
        return collect_metadata(data=data)

    def _collect_output_metadata(self, output):
        """Collect comprehensive metadata about the Output NumPy array."""
        return collect_output_metadata(output=output)

    def chat(self, query):
        """Handles user queries by generating and executing NumPy code."""
        assert isinstance(query, str)

        tries = 0
        while tries < self.MAX_TRIES:
            self.current_prompt = query
            _code = self.generate_numpy_code(query)

            if isinstance(_code, str):
                _res = self.execute_numpy_code(_code, self._data)
                if _res is None:
                    tries += 1
                    continue

                self._output_metadata = self._collect_output_metadata(_res)
                # Generate and run test code
                _testing_prompt = validate_llm_output(
                    query=query,
                    metadata=self.metadata,
                    output_metadata=self._output_metadata,
                )
                _testing_code = self.generate_llm_response(_testing_prompt)
                _testing_code = re.sub(r"```(\w+)?", "", _testing_code).strip()

                c.log(
                    f"""The following code will be executed as validation/test:
                    {_testing_code}
                """
                )

                _test_args = {"arr": self._data, "code_out": _res}
                _test_response = self.execute_numpy_code(_testing_code, _test_args)

                # Fix the boolean check with proper handling for arrays
                if _test_response is not None:
                    if isinstance(_test_response, bool):
                        if _test_response:
                            return _res
                    elif isinstance(_test_response, np.ndarray):
                        # Handle array truth value - use all() or any() based on your validation needs
                        if _test_response.size == 1:
                            if bool(_test_response.item()):
                                return _res
                        elif (
                            _test_response.all()
                        ):  # or .any() depending on validation requirements
                            return _res
                    else:
                        # For other non-None return types, evaluate as boolean
                        try:
                            if bool(_test_response):
                                return _res
                        except (ValueError, TypeError):
                            # If boolean conversion fails, consider it a failed test
                            pass

            tries += 1

        raise Exception(
            "The LLM failed to generate the correct response after maximum retries."
        )

    def generate_numpy_code(self, query):
        """Generate valid NumPy code from the query."""
        pr = self.generate_numpy_prompt(query)
        llm_res = self.generate_llm_response(pr)
        c.log(f"llm response is: \n {llm_res}")
        return self.assert_is_code(llm_res)

    def generate_llm_response(self, prompt):
        """Get LLM-generated response."""
        return self._code_generator.generate_response(prompt)

    def assert_is_code(self, llm_response):
        """Ensure LLM response is valid Python/NumPy code."""
        if not isinstance(llm_response, str):
            raise ValueError("LLM response is not a string")

        tries = 0
        while tries < self.MAX_TRIES:
            code = re.sub(r"```(\w+)?", "", llm_response).strip()
            try:
                if self._validator.validate_code(code):
                    c.log(f"The following code will be executed:\n {code}")
                    return code
            except SyntaxError:
                tries += 1
                llm_response = self.generate_llm_response(self.current_prompt)
                continue

            tries += 1

        raise Exception("Error Generating Valid LLM Response.")

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
            return local_vars.get("output")

        except Exception as e:
            c.log(f"Error executing code: {str(e)}")
            return None

    def generate_numpy_prompt(self, query):
        """Format the user query into a prompt for code generation."""
        return numpy_single_array_llm_prompt(query=query, metadata=self.metadata)
