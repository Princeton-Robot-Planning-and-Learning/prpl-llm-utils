# PRPL LLM Utils

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-llm-utils/actions/workflows/ci.yml/badge.svg)

LLM utilities from the Princeton Robot Planning and Learning group.

The main feature is the ability to save and load previous responses. There are a few other nice features, such as automatically retrying queries to deal with rate limiting or spotty internet.

## Usage Examples

```python
# Make sure OPENAI_API_KEY is set first.
from prpl_llm_utils.models import OpenAILLM
from pathlib import Path
llm = OpenAILLM("gpt-4o-mini", Path(".llm_cache"))
response, info = llm.query("Tell me a story")
# Querying again loads from cache.
assert llm.query("Tell me a story")[0] == response
# Querying with a different temperature or seed can change the response.
response2, _ = llm.query("Tell me a story", temperature=0.5)
response3, _ = llm.query("Tell me a story", seed=123)
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
