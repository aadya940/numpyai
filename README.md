### NumpyAI
A Natural Language Interface to the Numpy Library using LLMs.

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
