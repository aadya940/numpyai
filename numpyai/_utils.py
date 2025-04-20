import numpy as np
import re


class NumpyMetadataCollector:
    """
    A class to collect metadata from NumPy arrays and outputs of NumPy operations.
    """

    def __init__(self):
        pass

    def metadata(self, data):
        """Collect metadata about the given NumPy array."""
        md = {
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
                md["has_nan"] = bool(np.isnan(data).any())
                md["has_inf"] = bool(np.isinf(data).any())
                if data.size > 0 and not np.all(np.isnan(data)):
                    md["min"] = float(np.nanmin(data))
                    md["max"] = float(np.nanmax(data))
            except Exception:
                pass

        # Summary stats for smaller arrays
        if (
            data.size > 0
            and data.size <= 10000
            and np.issubdtype(data.dtype, np.number)
        ):
            try:
                md["zeros_count"] = int(np.count_nonzero(data == 0))
                md["non_zeros_count"] = int(np.count_nonzero(data))
            except Exception:
                pass

        try:
            md.update(
                {
                    "array-preview": (
                        data[: max(len(data) // 2, 15)] if (15 < len(data)) else data
                    ),
                }
            )
        except:
            pass

        # Large array hints
        if data.size > 1000000:
            md["large_array"] = True

        return md

    @staticmethod
    def collect_output_metadata(output):
        """Collect metadata about a NumPy operation output."""
        metadata = {"type": type(output).__name__}

        if isinstance(output, np.ndarray):
            metadata.update(
                {
                    "shape": output.shape,
                    "ndim": output.ndim,
                    "size": output.size,
                    "dtype": str(output.dtype),
                    "memory_size": output.nbytes,
                    "is_contiguous": output.flags.contiguous,
                    "is_fortran": output.flags.f_contiguous,
                    "has_nan": (
                        np.isnan(output).any()
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "has_inf": (
                        np.isinf(output).any()
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "is_structured": np.issubdtype(output.dtype, np.void),
                }
            )

            # Statistics
            try:
                metadata.update(
                    {
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                    }
                )
            except (TypeError, ValueError):
                pass

            # Sample elements
            if output.size > 0:
                sample_size = min(5, output.size)
                metadata["first_elements"] = output.flatten()[:sample_size].tolist()
                if output.size > sample_size * 2:
                    metadata["last_elements"] = output.flatten()[-sample_size:].tolist()

            # Large array hints
            if output.size > 1000000:
                metadata["large_array"] = True
            if not output.flags.contiguous and not output.flags.f_contiguous:
                metadata["non_contiguous"] = True

        elif np.isscalar(output):
            metadata["value"] = output
            if hasattr(output, "dtype"):
                metadata["dtype"] = str(output.dtype)

        elif isinstance(output, (list, tuple)):
            metadata.update(
                {
                    "length": len(output),
                    "sample": output[:5] if len(output) > 5 else output,
                }
            )

        elif output is None:
            metadata["is_none"] = True

        elif isinstance(output, str):
            metadata.update(
                {
                    "length": len(output),
                    "preview": output[:100] + "..." if len(output) > 100 else output,
                }
            )

        return metadata


def clean_code(code):
    """Strip the code from ```python ... ``` returned by the LLM"""
    return re.sub(r"```(\w+)?", "", code).strip()
