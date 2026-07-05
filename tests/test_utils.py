"""Tests for pure helpers that don't need an LLM."""

from __future__ import annotations

import numpy as np

from numpyai._utils import NumpyMetadataCollector, clean_code
from numpyai._validator import NumpyValidator


def test_clean_code_strips_fences():
    src = "```python\nprint('hi')\n```"
    assert clean_code(src) == "print('hi')"


def test_clean_code_no_fence_is_noop():
    assert clean_code("x = 1") == "x = 1"


def test_metadata_1d():
    arr = np.array([1.0, 2.0, np.nan])
    md = NumpyMetadataCollector().metadata(arr)
    assert md["shape"] == (3,)
    assert md["dims"] == 1
    assert md["has_nan"] is True


def test_metadata_scalar_zero_dim_does_not_crash():
    arr = np.array(5)
    md = NumpyMetadataCollector().metadata(arr)
    assert md["dims"] == 0
    assert md["shape"] == ()


def test_output_metadata_ndarray():
    md = NumpyMetadataCollector.collect_output_metadata(np.array([1, 2, 3]))
    assert md["type"] == "ndarray"
    assert md["shape"] == (3,)
    assert md["first_elements"] == [1, 2, 3]


def test_output_metadata_none():
    assert NumpyMetadataCollector.collect_output_metadata(None)["is_none"] is True


def test_validator_accepts_good_code():
    assert NumpyValidator().validate_code("output = 1 + 2") is True


def test_validator_rejects_bad_code():
    assert NumpyValidator().validate_code("output = 1 +") is False
