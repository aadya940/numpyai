### NumpyAI
A Natural Language Interface for NumPy powered by LLMs.

### About NumpyAI
NumpyAI enables seamless interaction with NumPy using natural language queries, making numerical computing more intuitive and efficient.

#### Key Features:
- **Natural Language Processing**: Convert plain language instructions into executable NumPy code.
- **Validation & Safety**: Automatically validates and tests generated code for correctness and security.
- **Transparency**: Logs all generated code and applies checks to ensure accuracy before execution.
- **History Tracking**: Keeps track of past operations for better reproducibility.
- **Undo/Redo Functionality**: Easily revert or reapply operations on NumPy arrays.

### Installation
```sh
pip install numpyai
```

### Usage Example
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

### Supported LLM Vendors
- Google Gemini

### Future Enhancements
- Support for additional LLM providers (OpenAI, Anthropic, etc.)
- Expanded validation mechanisms for complex operations
- Interactive debugging and visualization tools
