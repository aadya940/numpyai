import numpy as np


def collect_metadata(data):
    _md = {
        "is_numpy": isinstance(data, np.ndarray),
        "dims": data.ndim,
        "shape": data.shape,
        "size": data.size,
        "element_type": data.dtype,
        "byte_size": data.nbytes,
    }

    # Safe numeric data properties
    if np.issubdtype(data.dtype, np.number):
        try:
            _md["has_nan"] = bool(np.isnan(data).any())
            _md["has_inf"] = bool(np.isinf(data).any())

            # Basic range info (safely handles any dimension)
            if data.size > 0 and not np.all(np.isnan(data)):
                _md["min"] = float(np.nanmin(data))
                _md["max"] = float(np.nanmax(data))
        except Exception:
            pass  # Skip if calculations fail

    # For small to medium arrays, add useful summary stats
    if data.size > 0 and data.size <= 10000 and np.issubdtype(data.dtype, np.number):
        try:
            _md["zeros_count"] = int(np.count_nonzero(data == 0))
            _md["non_zeros_count"] = int(np.count_nonzero(data))
        except Exception:
            pass

    return _md


def collect_output_metadata(output):
    """
    Collect comprehensive metadata about the output from NumPy operations.

    Args:
        output: The result of a NumPy operation, could be a NumPy array, scalar, or other type

    Returns:
        dict: A dictionary containing metadata about the output
    """
    metadata = {}

    # Basic type information
    metadata["type"] = type(output).__name__

    # Handle different output types differently
    if isinstance(output, np.ndarray):
        # Array-specific metadata
        metadata["shape"] = output.shape
        metadata["ndim"] = output.ndim
        metadata["size"] = output.size
        metadata["dtype"] = str(output.dtype)

        # Statistical information (where applicable)
        try:
            metadata["min"] = float(output.min())
            metadata["max"] = float(output.max())
            metadata["mean"] = float(output.mean())
            metadata["std"] = float(output.std())
        except (TypeError, ValueError):
            # Skip statistical info for non-numeric arrays
            pass

        # Sample data (first few and last few elements)
        if output.size > 0:
            sample_size = min(5, output.size)
            metadata["first_elements"] = output.flatten()[:sample_size].tolist()
            if output.size > sample_size * 2:
                metadata["last_elements"] = output.flatten()[-sample_size:].tolist()

        # Check for special characteristics
        metadata["has_nan"] = (
            np.isnan(output).any() if np.issubdtype(output.dtype, np.number) else False
        )
        metadata["has_inf"] = (
            np.isinf(output).any() if np.issubdtype(output.dtype, np.number) else False
        )
        metadata["is_structured"] = np.issubdtype(output.dtype, np.void)

    elif np.isscalar(output):
        # Scalar-specific metadata
        metadata["value"] = output
        if hasattr(output, "dtype"):
            metadata["dtype"] = str(output.dtype)

    elif isinstance(output, (list, tuple)):
        # Collection-specific metadata
        metadata["length"] = len(output)
        metadata["sample"] = output[:5] if len(output) > 5 else output

    elif output is None:
        metadata["is_none"] = True

    elif isinstance(output, str):
        metadata["length"] = len(output)
        metadata["preview"] = output[:100] + "..." if len(output) > 100 else output

    # Performance hints for future operations
    if isinstance(output, np.ndarray):
        metadata["memory_size"] = output.nbytes
        metadata["is_contiguous"] = output.flags.contiguous
        metadata["is_fortran"] = output.flags.f_contiguous

        # Add hints about potential optimizations
        if output.size > 1000000:
            metadata["large_array"] = True
        if not output.flags.contiguous and not output.flags.f_contiguous:
            metadata["non_contiguous"] = True

    return metadata
