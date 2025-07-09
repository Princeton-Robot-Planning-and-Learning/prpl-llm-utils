"""Data structure and methods for code synthesis."""

import ast
import importlib
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from prpl_llm_utils.reprompting import RepromptCheck, create_reprompt_from_error_message
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
