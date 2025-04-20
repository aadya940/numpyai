import ast
import numpy as np


class NumpyValidator:
    def __init__(self) -> None:
        """A class that validates if the given code is correct numpy code."""
        pass

    def validate_code(self, code: str) -> bool:
        """Validates if the given code is syntactically correct and has valid NumPy signatures."""
        try:
            ast.parse(code)  # Check for syntax correctness
        except SyntaxError:
            return False

        return True

    def generate_validation_prompt(self, query, metadata, output_metadata):
        return f"""Generate Numpy Code to independently validate that the following output is 
        correct for the given query. The goal is to ensure correctness without simply 
        re-executing the same operation.
        
        Query: 
        {query}

        CRITICAL INSTRUCTIONS:
        1. The array is ALREADY defined as 'arr'. DO NOT redefine it (e.g., 'arr = ...').
        2. The output is ALREADY stored in 'code_out'. DO NOT redefine it (e.g., 'code_out = ...').
        3. Only import NumPy (which is already available). DO NOT import any other libraries.
        4. Use an independent verification method when possible instead of simply recomputing the query.
        5. Prioritize property-based checks (e.g., shape, statistics, known mathematical properties).
        6. Ensure proper handling of NaNs, infinities, and edge cases.
        7. Assign the final validation result to 'output' (Boolean: True if valid, False otherwise).
        8. DO NOT use 'return' statements—always assign the result to 'output'.
        9. Allow numerical tolerance (rtol=1e-5) where appropriate.
        10. The variable 'output' should be assigned only once if all test cases pass.
        11. If the type of the output is `None`, don't test anything.

        The input array has these properties:
        {metadata}
        
        The expected output has these properties:
        {output_metadata}

        CORRECT EXAMPLES:
        
        # Example: Verify sum of elements is correct (using alternative approach)
        expected_sum = np.sum(arr)
        output = np.isclose(expected_sum, code_out)
        print(f"Expected sum: {{expected_sum}}, Got: {{code_out}}, Valid: {{output}}")
        
        # Example: Validate correlation matrix without directly recomputing
        if arr.ndim == 2:
            diag_ones = np.allclose(np.diag(code_out), 1.0, rtol=1e-5)  # Correlation matrices have 1s on diagonal
            symmetry = np.allclose(code_out, code_out.T, rtol=1e-5)  # Correlation matrices should be symmetric
            output = diag_ones and symmetry
        else:
            output = False
        print(f"Validation: Diagonal ones: {{diag_ones}}, Symmetry: {{symmetry}}, Valid: {{output}}")

        # Example: Checking sorted array without sorting it again
        sorted_check = np.all(code_out[:-1] <= code_out[1:])
        output = sorted_check
        print(f"Sorted correctly: {{sorted_check}}, Valid: {{output}}")
        
        INCORRECT EXAMPLES (DO NOT DO THIS):
        # DON'T recompute the same function
        expected = np.corrcoef(arr, rowvar=False)  # Avoid directly recomputing
        output = np.allclose(expected, code_out)  

        # DON'T redefine 'arr' or 'code_out'
        arr = np.array([...])
        code_out = np.array([...])
        
        # DON'T use return statements
        return np.allclose(arr * 2, code_out)
        
        # DON'T import external libraries
        import scipy.stats  # Not allowed
        """

    def generate_validation_prompt_multiple(
        self, query, input_metadata, output_metadata, error=None
    ):
        """New method for multiple arrays validation."""
        input_metadata_str = "\n".join(
            f"- **{name}**: {metadata}" for name, metadata in input_metadata.items()
        )

        __prompt = f"""Generate NumPy Code to independently validate that the following output is 
        correct for the given query. The goal is to ensure correctness without simply 
        re-executing the same operation.
        
        Query: 
        {query}

        CRITICAL INSTRUCTIONS:
        1. The arrays are ALREADY defined as 'arr1', 'arr2', etc. DO NOT redefine them.
        2. The output is ALREADY stored in 'code_out'. DO NOT redefine it.
        3. Only import NumPy (which is already available). DO NOT import any other libraries.
        4. Use an independent verification method when possible instead of simply recomputing the query.
        5. Prioritize property-based checks (e.g., shape, statistics, known mathematical properties).
        6. Ensure proper handling of NaNs, infinities, and edge cases.
        7. Assign the final validation result to 'output' (Boolean: True if valid, False otherwise).
        8. DO NOT use 'return' statements—always assign the result to 'output'.
        9. Allow numerical tolerance (rtol=1e-5) where appropriate.
        10. The variable 'output' should be assigned only once if all test cases pass.
        11. If the type of the output is `None`, don't test anything.
        
        The input arrays have these properties:
        {input_metadata_str}
        
        The expected output has these properties:
        {output_metadata}

        """
        if error:
            __prompt += f"Please take care of the following error message:\n {error}"
        return __prompt
