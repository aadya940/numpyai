<p align="center">
<img src="https://github.com/user-attachments/assets/7d6244d2-2a94-42c7-99e1-ba2953c21781" alt="logo" width="500">
</p>


### NumpyAI
A Natural Language Interface for [NumPy](https://github.com/numpy/numpy) powered by LLMs. Empowering mindful data analysis using Generative AI.

### About NumpyAI
NumpyAI enables seamless interaction with NumPy using natural language queries, making numerical computing more intuitive and efficient.

#### Key Features:
- Writes NumPy code for you based on your natural language queries.
- Know what data-analysis steps to apply on your data using `numpyai.Diagnosis`.
- Talk to multiple arrays using `numpyai.NumpyAISession`.
- Checks the validity of the generated code.
- Unit tests the code before returning the final-output.
- Full transparency, know what code was executed by the LLM using the `verbose=True` flag.
- Supports frameworks like `sklearn` and `matplotlib` for basic tasks.
- Interactive debugging and re-tries.


### Installation
```sh
pip install numpyai
```

### Installation from Source

Clone the project then:

```sh
cd numpyai/; pip install -r requirements.txt ; pip install .
```


### Setup

Windows
```sh
set GOOGLE_API_KEY=...
```

Linux
```sh
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

#### Diagnosis
```python
import numpyai as npi
import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.random.random((2, 3))

sess = npi.NumpyAISession([arr1, arr2])
diag = npi.Diagnosis(sess)
print(diag.steps(task="Tell me the exact and pithy steps to analyse and select which ML model to use for this data. There should be no more than 7 steps"))
```

### Supported LLM Vendors
- Google Gemini
- OpenAI
- Anthropic

### Contributing Guidelines
- Apply the `black` formatter.
- The code should be well documented and be rendered in the docs.
- For testing, add it in the `examples/all_functionality.ipynb` notebook.
- Ensure backward compatibility.

Thank you and looking forward to seeing you contribute to NumpyAI :) !
