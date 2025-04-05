<img src="https://github.com/user-attachments/assets/7d6244d2-2a94-42c7-99e1-ba2953c21781" alt="logo" width="150">



### NumpyAI
A Natural Language Interface for NumPy powered by LLMs.

### About NumpyAI
NumpyAI enables seamless interaction with NumPy using natural language queries, making numerical computing more intuitive and efficient.

#### Key Features:
- **Natural Language Processing**: Convert plain language instructions into executable NumPy code.
- **Validation & Safety**: Automatically validates and tests generated code for correctness and security.
- **Transparency**: Logs all generated code and applies checks to ensure accuracy before execution.
- We don't allow AI to reassign or change the internal arrays passed to `numpyai.array` or `numpyai.NumpyAISession`. We believe they need to be intentional decisions by the user.


### Installation
```sh
pip install numpyai
```

Windows
```
set GOOGLE_API_KEY=...
```

Linux
```
export GOOGLE_API_KEY=...
```

### Usage Example

#### Single Array
```python
import numpyai as npi
import numpy as np

# Ensure GOOGLE_API_KEY environment variable is set.

# Create an array instance
data = [[1, 2, 3, 4, 5, np.nan], [np.nan, 3, 5, 3.1415, 2, 2]]
arr = npi.array(data)

# Query NumPyAI with natural language
print(arr.chat("Compute the height and width of the image using NumPy."))  # Expected output: (2, 6)
```

#### Multiple Arrays
```python
import numpyai as npi
import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.random.random((2, 3))

sess = npi.NumpyAISession([arr1, arr2])
imputed_array = sess.chat("Impute the first array with the mean of the second array.")
```

### Supported LLM Vendors
- Google Gemini

### Future Enhancements
- Support for additional LLM providers (OpenAI, Anthropic, etc.)
- Expanded validation mechanisms for complex operations
- Interactive debugging and visualization tools
