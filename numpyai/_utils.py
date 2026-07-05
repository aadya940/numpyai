"""Utility helpers: metadata collection and code cleanup."""

from __future__ import annotations

import re

import numpy as np


class NumpyMetadataCollector:
    """Collect metadata from NumPy arrays and NumPy-operation outputs."""

    def metadata(self, data: np.ndarray) -> dict:
        """Collect metadata about the given NumPy array."""
        md: dict = {
            "is_numpy": isinstance(data, np.ndarray),
            "dims": data.ndim,
            "shape": data.shape,
            "size": data.size,
            "element_type": data.dtype,
            "byte_size": data.nbytes,
        }

        if np.issubdtype(data.dtype, np.number):
            try:
                md["has_nan"] = bool(np.isnan(data).any())
                md["has_inf"] = bool(np.isinf(data).any())
                if data.size > 0 and not np.all(np.isnan(data)):
                    md["min"] = float(np.nanmin(data))
                    md["max"] = float(np.nanmax(data))
            except (TypeError, ValueError):
                pass

        if data.size > 0 and data.size <= 10_000 and np.issubdtype(data.dtype, np.number):
            try:
                md["zeros_count"] = int(np.count_nonzero(data == 0))
                md["non_zeros_count"] = int(np.count_nonzero(data))
            except (TypeError, ValueError):
                pass

        if data.ndim >= 1:
            try:
                preview_len = max(len(data) // 2, 15)
                md["array-preview"] = data[:preview_len] if len(data) > 15 else data
            except TypeError:
                pass

        if data.size > 1_000_000:
            md["large_array"] = True

        return md

    @staticmethod
    def collect_output_metadata(output) -> dict:
        """Collect metadata about a NumPy operation output."""
        metadata: dict = {"type": type(output).__name__}

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
                        bool(np.isnan(output).any())
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "has_inf": (
                        bool(np.isinf(output).any())
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "is_structured": np.issubdtype(output.dtype, np.void),
                }
            )

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

            if output.size > 0:
                sample_size = min(5, output.size)
                metadata["first_elements"] = output.flatten()[:sample_size].tolist()
                if output.size > sample_size * 2:
                    metadata["last_elements"] = output.flatten()[-sample_size:].tolist()

            if output.size > 1_000_000:
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


_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$", re.MULTILINE)


def clean_code(code: str) -> str:
    """Strip markdown code fences from an LLM response."""
    return _FENCE_RE.sub("", code).strip()
