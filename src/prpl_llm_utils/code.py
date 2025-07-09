"""Data structure and methods for code synthesis."""

import ast
import importlib
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import (
    RepromptCheck,
    create_reprompt_from_error_message,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response


@dataclass(frozen=True)
class SynthesizedPythonFunction:
    """Wraps a piece of Python code that contains a function with a given name.

    The typical flow is that an LLM outputs the code as a string, then
    we create one of these class instances, then invoke the function by
    calling run().
    """

    function_name: str
    code_str: str

    @cached_property
    def filepath(self) -> Path:
        """Get a file with the code string implemented in it."""
        filename = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".py").name)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.code_str)
        return filename

    def _load_module(self) -> Any:
        module_name = f"{self.filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.filepath)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        assert module is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def run(self, *input_args: Any) -> Any:
        """Run the function on an input (that will be unpacked)."""
        module = self._load_module()
        fn = getattr(module, self.function_name)
        return fn(*input_args)


class SyntaxRepromptCheck(RepromptCheck):
    """Check the syntax of a response."""

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        python_code = parse_python_code_from_text(response.text)
        if python_code is None:
            error_msg = "No python code was found in the response."
        else:
            try:
                ast.parse(python_code)
                return None
            except SyntaxError as e:
                error_msg = "\n".join(traceback.format_exception(e))
        return create_reprompt_from_error_message(query, response, error_msg)


class FunctionOutputRepromptCheck(RepromptCheck):
    """Check whether the synthesized Python function produces valid output.

    It is up to the user of this class how "valid" is defined.
    """

    def __init__(
        self,
        function_name: str,
        inputs: list[Any],
        output_check_fns: list[Callable[[Any], bool]],
    ) -> None:
        assert len(inputs) == len(
            output_check_fns
        ), "Expecting one check function per input"
        self._function_name = function_name
        self._inputs = inputs
        self._output_check_fns = output_check_fns

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        python_code = parse_python_code_from_text(response.text)
        assert python_code is not None  # should be checked first with syntax
        fn = SynthesizedPythonFunction(self._function_name, python_code)
        for fn_in, check_fn in zip(self._inputs, self._output_check_fns, strict=True):
            fn_out = fn.run(*fn_in)
            if not check_fn(fn_out):
                error_msg = (
                    f"Given the input {fn_in}, the output of {self._function_name} "
                    f"was {fn_out}, which is invalid"
                )
                return create_reprompt_from_error_message(query, response, error_msg)
        return None


def parse_python_code_from_text(text: str) -> str | None:
    """Parse Python code from text, assuming ```python tag."""
    # Parse out python code if it exists.
    python_code_prefix = "```python"
    if python_code_prefix in text:
        python_start = text.index(python_code_prefix)
        python_remainder = text[python_start + len(python_code_prefix) :]
        if "```" in python_remainder:
            python_end = python_remainder.index("```")
        else:
            python_end = len(python_remainder)
        python_response = python_remainder[:python_end]
        return python_response
    return None


def synthesize_python_function_with_llm(
    function_name: str,
    model: PretrainedLargeModel,
    query: Query,
    reprompt_checks: list[RepromptCheck] | None = None,
    max_attempts: int = 5,
) -> SynthesizedPythonFunction:
    """Synthesize a Python function with an LLM."""
    if reprompt_checks is None:
        reprompt_checks = []
    response = query_with_reprompts(model, query, reprompt_checks, max_attempts)
    python_code = parse_python_code_from_text(response.text)
    if python_code is None:
        raise RuntimeError("No python code found. Consider SyntaxRepromptCheck().")
    return SynthesizedPythonFunction(function_name, python_code)
