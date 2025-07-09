"""Data structure and methods for code synthesis."""

import importlib
import sys
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any


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

    def run(self, input_args: list[Any]) -> Any:
        """Run the function on an input (that will be unpacked)."""
        module = self._load_module()
        fn = getattr(module, self.function_name)
        return fn(*input_args)  # type: ignore
