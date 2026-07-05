"""Syntactic validation of generated NumPy code.

The library's *semantic* validation runs through the judgment agent
(see :class:`numpyai.Judgment`). This module only verifies that the
generated code parses.
"""

from __future__ import annotations

import ast


class NumpyValidator:
    """Verify that a generated code snippet is syntactically valid Python."""

    def validate_code(self, code: str) -> bool:
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True
