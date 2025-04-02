import numpy as np
import re
from rich.console import Console
from typing import Any, Dict, List, Optional, Union, Tuple

from ._validator import NumpyValidator
from ._ai import NumpyCodeGen
from ._utils import collect_metadata, collect_output_metadata
from ._prompts import numpy_single_array_llm_prompt, validate_llm_output

c = Console()

c.log("Ensure your API KEY is set for your LLM as an environment variable.")


class array:
    def __init__(self, data):
        """NumpyAI wrapper for `numpy.ndarray`."""
        assert isinstance(data, np.ndarray)
        self._data = data

        self.metadata = self._collect_metadata(self._data)
        self.current_prompt = None
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen()
        self.MAX_TRIES = 3

        self._output_metadata = {}

        # Operation history for undo functionality
        self._history = [
            np.array(data).copy()
        ]  # Initialize with a copy of initial data
        self._history_index = 0  # Start at the first element
        self._max_history = 50  # Maximum number of operations to store

    # Basic operators with scalar support
    def __add__(self, other):
        if isinstance(other, array):
            result = self._data + other._data
        else:
            result = self._data + other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __radd__(self, other):
        result = other + self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __sub__(self, other):
        if isinstance(other, array):
            result = self._data - other._data
        else:
            result = self._data - other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rsub__(self, other):
        result = other - self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __mul__(self, other):
        if isinstance(other, array):
            result = self._data * other._data
        else:
            result = self._data * other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rmul__(self, other):
        result = other * self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __truediv__(self, other):
        if isinstance(other, array):
            result = self._data / other._data
        else:
            result = self._data / other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rtruediv__(self, other):
        result = other / self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __floordiv__(self, other):
        if isinstance(other, array):
            result = self._data // other._data
        else:
            result = self._data // other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rfloordiv__(self, other):
        result = other // self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __pow__(self, other):
        if isinstance(other, array):
            result = self._data**other._data
        else:
            result = self._data**other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rpow__(self, other):
        result = other**self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __mod__(self, other):
        if isinstance(other, array):
            result = self._data % other._data
        else:
            result = self._data % other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rmod__(self, other):
        result = other % self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        if isinstance(other, array):
            result = self._data @ other._data
        else:
            result = self._data @ other
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __rmatmul__(self, other):
        result = other @ self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __neg__(self):
        result = -self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __pos__(self):
        result = +self._data
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __abs__(self):
        result = abs(self._data)
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __getitem__(self, index):
        result = self._data[index]
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def __setitem__(self, index, value):
        old_data = self._data.copy()
        self._data[index] = value
        self._add_to_history(old_data)  # Store the previous state

    def __repr__(self):
        return f"numpyai.array({repr(self._data)})"

    # Common NumPy functions
    def sum(self, axis=None, keepdims=False):
        """Sum of array elements over a given axis."""
        result = self._data.sum(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def mean(self, axis=None, keepdims=False):
        """Mean of array elements over a given axis."""
        result = self._data.mean(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def std(self, axis=None, keepdims=False):
        """Standard deviation of array elements over a given axis."""
        result = self._data.std(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def min(self, axis=None, keepdims=False):
        """Minimum of array elements over a given axis."""
        result = self._data.min(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def max(self, axis=None, keepdims=False):
        """Maximum of array elements over a given axis."""
        result = self._data.max(axis=axis, keepdims=keepdims)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def argmin(self, axis=None):
        """Indices of minimum values along an axis."""
        result = self._data.argmin(axis=axis)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def argmax(self, axis=None):
        """Indices of maximum values along an axis."""
        result = self._data.argmax(axis=axis)
        if isinstance(result, np.ndarray):
            new_arr = array(result)
            new_arr._history = self._history.copy()
            new_arr._history_index = self._history_index
            new_arr._add_to_history(result)
            return new_arr
        return result

    def clip(self, min=None, max=None):
        """Clip (limit) the values in the array."""
        result = self._data.clip(min=min, max=max)
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def round(self, decimals=0):
        """Round array to the given number of decimals."""
        result = np.round(self._data, decimals=decimals)
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def reshape(self, *shape):
        """Reshape the array."""
        result = self._data.reshape(*shape)
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    def transpose(self, *axes):
        """Transpose the array."""
        result = self._data.transpose(*axes)
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

    # Properties
    @property
    def T(self):
        """Return the transpose of the array."""
        result = self._data.T
        new_arr = array(result)
        new_arr._history = self._history.copy()
        new_arr._history_index = self._history_index
        new_arr._add_to_history(result)
        return new_arr

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
        self._add_to_history(self._data)  # Store the previous state
        self._data = new_array

    # History management methods
    def _add_to_history(self, result):
        """Add an operation result to history."""
        # Remove any future history if we're in the middle of the history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        # Add current state to history
        self._history.append(np.array(result).copy())
        self._history_index = len(self._history) - 1

        # Trim history if it exceeds max_history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
            self._history_index = len(self._history) - 1

    def undo(self):
        """Restore the previous state of the array."""
        if self._history_index > 0:
            self._history_index -= 1
            self._data = self._history[self._history_index].copy()
            return self
        else:
            raise IndexError("No operations to undo.")

    def redo(self):
        """Redo the last undone operation."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._data = self._history[self._history_index].copy()
            return self
        else:
            raise IndexError("No operations to redo.")

    def history_size(self):
        """Return the number of operations in history."""
        return len(self._history)

    def clear_history(self):
        """Clear operation history except current state."""
        current_state = self._data.copy()
        self._history = [current_state]
        self._history_index = 0

    def set_max_history(self, size):
        """Set the maximum number of operations to store in history."""
        if size < 1:
            raise ValueError("History size must be at least 1")
        self._max_history = size
        if len(self._history) > size:
            # Keep the current state and trim history
            self._history = self._history[-size:]
            self._history_index = len(self._history) - 1

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
                                # Add result to history if it's an ndarray
                                if isinstance(_res, np.ndarray):
                                    self._add_to_history(_res)
                                    return _res
                                return _res
                        elif isinstance(_test_response, np.ndarray):
                            # Handle array truth value - use all() or any() based on your validation needs
                            if _test_response.size == 1:
                                if bool(_test_response.item()):
                                    if isinstance(_res, np.ndarray):
                                        self._add_to_history(_res)
                                        return _res
                                    return _res
                            elif (
                                _test_response.all()
                            ):  # or .any() depending on validation requirements
                                if isinstance(_res, np.ndarray):
                                    self._add_to_history(_res)
                                    return _res
                                return _res
                        else:
                            # For other non-None return types, evaluate as boolean
                            try:
                                if bool(_test_response):
                                    if isinstance(_res, np.ndarray):
                                        self._add_to_history(_res)
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
                llm_response = self.generate_llm_response(self.current_prompt)
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

    def generate_numpy_prompt(self, query):
        """Format the user query into a prompt for code generation."""
        return numpy_single_array_llm_prompt(query=query, metadata=self.metadata)
