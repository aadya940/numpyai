import numpy as np
import re
from rich.console import Console
from typing import Any, Dict, List, Optional, Union, Tuple

from ._validator import NumpyValidator
from ._ai import NumpyCodeGen
from ._utils import NumpyMetadataCollector
from ._exceptions import NumpyAIError

c = Console()

c.log("Ensure your API KEY is set for your LLM as an environment variable.")


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

    # Common NumPy functions
    def sum(self, axis=None, keepdims=False):
        """Sum of array elements over a given axis."""
        result = self._data.sum(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def mean(self, axis=None, keepdims=False):
        """Mean of array elements over a given axis."""
        result = self._data.mean(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def std(self, axis=None, keepdims=False):
        """Standard deviation of array elements over a given axis."""
        result = self._data.std(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def min(self, axis=None, keepdims=False):
        """Minimum of array elements over a given axis."""
        result = self._data.min(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def max(self, axis=None, keepdims=False):
        """Maximum of array elements over a given axis."""
        result = self._data.max(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def argmin(self, axis=None):
        """Indices of minimum values along an axis."""
        result = self._data.argmin(axis=axis)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def argmax(self, axis=None):
        """Indices of maximum values along an axis."""
        result = self._data.argmax(axis=axis)
        if isinstance(result, np.ndarray):
            return array(result)
        return result

    def clip(self, min=None, max=None):
        """Clip (limit) the values in the array."""
        result = self._data.clip(min=min, max=max)
        return array(result)

    def round(self, decimals=0):
        """Round array to the given number of decimals."""
        result = np.round(self._data, decimals=decimals)
        return array(result)

    def reshape(self, *shape):
        """Reshape the array."""
        result = self._data.reshape(*shape)
        return array(result)

    def transpose(self, *axes):
        """Transpose the array."""
        result = self._data.transpose(*axes)
        return array(result)

    # Properties
    @property
    def T(self):
        """Return the transpose of the array."""
        result = self._data.T
        return array(result)

    @property
    def shape(self):
        """Return the shape of the array."""
        return self._data.shape

    @property
    def ndim(self):
        """Return the number of dimensions of the array."""
        return self._data.ndim

    @property
    def size(self):
        """Return the size of the array."""
        return self._data.size

    @property
    def dtype(self):
        """Return the data type of the array."""
        return self._data.dtype

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

        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            self.current_prompt = query
            try:
                _code = self.generate_numpy_code(query)
                if isinstance(_code, str):
                    _res = self.execute_numpy_code(_code, self._data)
                    if _res is None:
                        error_messages.append(
                            f"Try {tries+1}: Code execution returned None"
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
                                if isinstance(_res, np.ndarray):
                                    return _res
                                return _res
                        elif isinstance(_test_response, np.ndarray):
                            # Handle array truth value - use all() or any() based on your validation needs
                            if _test_response.size == 1:
                                if bool(_test_response.item()):
                                    if isinstance(_res, np.ndarray):
                                        return _res
                                    return _res
                            elif (
                                _test_response.all()
                            ):  # or .any() depending on validation requirements
                                if isinstance(_res, np.ndarray):
                                    return _res
                                return _res
                        else:
                            # For other non-None return types, evaluate as boolean
                            try:
                                if bool(_test_response):
                                    if isinstance(_res, np.ndarray):
                                        return _res
                                    return _res
                            except (ValueError, TypeError):
                                # If boolean conversion fails, consider it a failed test
                                error_messages.append(
                                    f"Try {tries+1}: Validation failed - couldn't convert test response to boolean"
                                )
            except Exception as e:
                error_messages.append(f"Try {tries+1}: {str(e)}")

            tries += 1

        # More detailed error message with history of what went wrong
        raise NumpyAIError(
            f"Failed to generate correct response after {self.MAX_TRIES} attempts. "
            f"Error details: {'; '.join(error_messages)}"
        )

    def generate_numpy_code(self, query):
        """Generate valid NumPy code from the query."""
        pr = self._code_generator.generate_llm_prompt(
            query=query, metadata=self.metadata
        )
        llm_res = self._code_generator.generate_response(pr)
        c.log(f"llm response is: \n {llm_res}")
        return self.assert_is_code(llm_res)

    def assert_is_code(self, llm_response):
        """Ensure LLM response is valid Python/NumPy code."""
        if not isinstance(llm_response, str):
            raise ValueError("LLM response is not a string")

        tries = 0
        error_messages = []

        while tries < self.MAX_TRIES:
            code = re.sub(r"```(\w+)?", "", llm_response).strip()
            try:
                if self._validator.validate_code(code):
                    c.log(f"The following code will be executed:\n {code}")
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
            f"Error generating valid code after {self.MAX_TRIES} attempts. "
            f"Error details: {'; '.join(error_messages)}"
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
            return local_vars.get("output")

        except Exception as e:
            c.log(f"Error executing code: {str(e)}")
            return None
