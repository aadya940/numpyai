### NumpyAI
A Natural Language Interface to the Numpy Library using LLMs.

### About NumpyAI

This library, NumpyAI, is a Python package designed to facilitate the generation and execution of NumPy code through natural language queries. 
- Natural Language Processing: Users can input queries in plain language, and the library translates these into executable NumPy code.
- Validation: The library includes mechanisms to validate the generated code, ensuring it adheres to NumPy's syntax and operational standards.
- Transparency: We tell you exactly what could the LLM executes through logging and apply tests on the output to validate the output if the output is True or not.

### Examples

```python
import numpyai as npi
import numpy as np

# Ensure GOOGLE_API_KEY environment variable is set.

# Create an array instance.
data = [[1, 2, 3, 4, 5, np.nan], [np.nan, 3, 5, 3.1415, 2, 2]]
arr = npi.array(data)

# Example queries.
print(
    arr.chat(
        "Compute the Height and Width of the image using NumPy."
    )
)  # Should return 3.0
```

### Supported LLM Vendors

- Google Gemini
