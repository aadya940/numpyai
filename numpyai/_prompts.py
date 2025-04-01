def numpy_single_array_llm_prompt(query, metadata):
    return f"""Generate NumPy code to perform the following operation: \n
        {query}. \n

        CRITICAL INSTRUCTIONS:
        1. The array is ALREADY defined as 'arr'. DO NOT create a new array with 'arr = ...'.
        2. DO NOT IMPORT any libraries except numpy (which is already imported).
        3. DO NOT use scipy, pandas, sklearn, or any other library. Use ONLY numpy functions.
        4. Return ONLY the code that operates on the existing 'arr' variable.
        5. There should always be exactly one variable named "output" which contains what 
        the user asked for.
        6. Print what is important so the code is explainable.
        7. Ensure data is properly cleaned before executing any code.
        
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


def validate_llm_output(query, metadata, output_metadata):
    return f"""Generate Numpy Code to validate that the following output is or can be
    the correct output for the following query.
    
    Query: 
    {query}


    CRITICAL INSTRUCTIONS:
    1. The array is ALREADY defined as 'arr'. DO NOT create a new array with 'arr = ...'.
    2. The output is ALREADY defined as `code_out`. DO NOT create a new variable with `code_out = ...`.
    3. DO NOT IMPORT any libraries except numpy (which is already imported).
    4. The validation logic should be simple and minimalistic.
    5. Print what is important so the code is explainable.
    6. Ensure data is properly cleaned before executing any code.
    7. Set a variable called 'output' to True if validation passes, else False.
    8. DO NOT use 'return' statements - assign the result to 'output' variable.
    9. Allow numerical tolerance (rtol=1e-5) where appropriate.


    The array has these properties:
    {metadata}
    
    The output has the following properties:
    {output_metadata}

    CORRECT EXAMPLES:
    # Replace NaN values with zero
    arr_copy = arr.copy()
    arr_copy[np.isnan(arr_copy)] = 0
    output = np.allclose(arr_copy * 2, code_out)
    print(f"Validation result: {{output}}")
    
    # Calculate mean of array
    result = np.mean(arr)
    output = np.isclose(result, code_out)
    print(f"Expected: {{result}}, Got: {{code_out}}, Valid: {{output}}")
    
    INCORRECT EXAMPLES (DO NOT DO THIS):
    # DON'T create a new array
    arr = np.array([1, 2, 3, 4, 5])
    code_out = np.array([2, 4, 6, 8, 10])
    
    # DON'T use return statements
    return np.array_equal(arr * 2, code_out)
    
    # DON'T import scipy or other libraries
    from scipy import stats
    result = stats.zscore(arr)
    """