"""Utility functions."""

import hashlib
from typing import Any


def consistent_hash(obj: Any) -> int:
    """A hash function that is consistent between sessions, unlike hash()."""
    obj_str = repr(obj)
    obj_bytes = obj_str.encode("utf-8")
    hash_hex = hashlib.sha256(obj_bytes).hexdigest()
    hash_int = int(hash_hex, 16)
    # Mimic Python's built-in hash() behavior by returning a 64-bit signed int.
    # This makes it comparable to hash()'s output range.
    return hash_int if hash_int < 2**63 else hash_int - 2**6


def parse_python_code_from_llm_response(response: str) -> str:
    """Parse Python code from an LLM response, assuming ```python tag."""
    # Parse out python code if it exists.
    python_code_prefix = "```python"
    if python_code_prefix in response:
        python_start = response.index(python_code_prefix)
        python_remainder = response[python_start + len(python_code_prefix) :]
        if "```" in python_remainder:
            python_end = python_remainder.index("```")
        else:
            python_end = len(python_remainder)
        python_response = python_remainder[:python_end]
        return python_response
    return response
