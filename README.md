# PRPL LLM Utils

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-llm-utils/actions/workflows/ci.yml/badge.svg)

LLM utilities from the Princeton Robot Planning and Learning group.

The main feature is the ability to save and load previous responses.

## Usage Examples

```python
# Make sure OPENAI_API_KEY is set first.
from prpl_llm_utils.models import OpenAIModel
from pathlib import Path
llm = OpenAIModel("gpt-4o-mini", Path(".llm_cache"))
response = llm.query("What's a funny one liner?")
# Querying again loads from cache.
assert llm.query("What's a funny one liner?").text == response.text
# Querying with different hyperparameters can change the response.
response2 = llm.query("What's a funny one liner?", hyperparameters={"seed": 123})
```

## Requirements

- Python 3.10+
- Tested on MacOS Monterey and Ubuntu 22.04

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.

## Acknowledgements

This code descends from [predicators](https://github.com/Learning-and-Intelligent-Systems/predicators) and includes contributions from a number of people, including especially Nishanth Kumar.
