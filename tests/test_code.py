"""Tests for code.py."""

import tempfile
from pathlib import Path

from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.code import (
    SyntaxRepromptCheck,
    SynthesizedPythonFunction,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OrderedResponseModel
from prpl_llm_utils.structs import Query, Response


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


def test_synthesize_python_function_with_llm():
    """Tests for synthesize_python_function_with_llm()."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    cache_path = Path(cache_dir.name)
    cache = FilePretrainedLargeModelCache(cache_path)

    reprompt_checks = [SyntaxRepromptCheck()]
    function_name = "count_good_dogs"
    input_output_examples = [([["nomsy", "rover"]], 2), ([["nomsy"]], 1)]

    query = Query(
        """Generate a Python function of the form
    
def count_good_dogs(dog_names: list[str]) -> int:
    # your code here
"""
    )

    response_with_syntax_error = Response(
        """```python
def count_good_dogs(dog_names: list[str) -> int:
    return len(dog_names)
```    
""",
        {},
    )

    response_with_correct_answer = Response(
        """```python
def count_good_dogs(dog_names: list[str]) -> int:
    return len(dog_names)
```    
""",
        {},
    )

    ordered_responses = [response_with_syntax_error, response_with_correct_answer]
    llm = OrderedResponseModel(ordered_responses, cache)

    synthesized_python_fn = synthesize_python_function_with_llm(
        function_name, llm, query, reprompt_checks=reprompt_checks
    )

    for input_args, expected_output in input_output_examples:
        assert synthesized_python_fn.run(*input_args) == expected_output
