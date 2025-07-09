"""Tests for code.py."""

from prpl_llm_utils.code import SynthesizedPythonFunction


def test_synthesized_python_function():
    """Tests for SynthesizedPythonFunction()."""

    code_str = """
from dataclasses import dataclass

@dataclass
class Dog:

    name: str
    is_cute: bool = True


def count_cute_dogs(dog_names: list[str]) -> int:
    dogs = [Dog(d) for d in dog_names]
    return sum(d.is_cute for d in dogs)
"""

    synthesized_python_fn = SynthesizedPythonFunction("count_cute_dogs", code_str)
    assert synthesized_python_fn.run(["nomsy"]) == 1
    assert synthesized_python_fn.run(["nomsy", "puddles"]) == 2


def test_code_synthesis():
    """Tests code synthesis."""
    # TODO
