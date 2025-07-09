"""Interfaces for large language models."""

import abc
import json
import logging
import os
from pathlib import Path
from typing import Hashable

import openai
import PIL.Image

from prpl_llm_utils.structs import Query, Response


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    def __init__(self, cache_dir: Path, use_cache_only: bool = False) -> None:
        self._cache_dir = cache_dir
        self._use_cache_only = use_cache_only

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this model.

        This identifier should include sufficient information so that
        querying the same model with the same query and same identifier
        should yield the same result.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _run_query(self, query: Query) -> Response:
        """This is the main method that subclasses must implement.

        This helper method is called by query(), which caches the
        queries and responses to disk.
        """
        raise NotImplementedError("Override me!")

    def query(
        self,
        prompt: str,
        imgs: list[PIL.Image.Image] | None = None,
        hyperparameters: dict[str, Hashable] | None = None,
    ) -> Response:
        """Sample one or more completions from a prompt; also return metadata.

        Responses are saved to disk.
        """
        # Create the query.
        query = Query(prompt, imgs=imgs, hyperparameters=hyperparameters)
        # Set up the cache.
        cache_dir = self._get_cache_dir(query)
        if not (cache_dir / "prompt.txt").exists():
            if self._use_cache_only:
                raise ValueError("No cached response found for prompt.")
            logging.debug(f"Querying model {self.get_id()} with new prompt.")
            # Query the model.
            response = self._run_query(query)
            # Cache the text prompt.
            prompt_file = cache_dir / "prompt.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            # Cache the image prompt if it exists.
            if imgs is not None:
                imgs_folderpath = cache_dir / "imgs"
                imgs_folderpath.mkdir(exist_ok=True)
                for i, img in enumerate(imgs):
                    filename_suffix = str(i) + ".jpg"
                    img.save(imgs_folderpath / filename_suffix)
            # Cache each response.
            completion_file = cache_dir / "completion.txt"
            with open(completion_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            # Cache the metadata.
            metadata_file = cache_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(response.metadata, f)
            logging.debug(f"Saved model response to {cache_dir}.")
        # Load the saved completions.
        completion_file = cache_dir / "completion.txt"
        with open(completion_file, "r", encoding="utf-8") as f:
            completion = f.read()
        # Load the metadata.
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Create the response.
        response = Response(completion, metadata)
        logging.debug(f"Loaded model response from {cache_dir}.")
        return response

    def _get_cache_dir(self, query: Query) -> Path:
        # Set up the cache directory.
        self._cache_dir.mkdir(exist_ok=True)
        model_id = self.get_id()
        query_id = query.get_id()
        cache_foldername = f"{model_id}_{query_id}"
        cache_folderpath = self._cache_dir / cache_foldername
        cache_folderpath.mkdir(exist_ok=True)
        return cache_folderpath


class OpenAIModel(PretrainedLargeModel):
    """Common interface with methods for all OpenAI-based models."""

    def __init__(
        self, model_name: str, cache_dir: Path, use_cache_only: bool = False
    ) -> None:
        self._model_name = model_name
        assert "OPENAI_API_KEY" in os.environ, "Need to set OPENAI_API_KEY"
        super().__init__(cache_dir, use_cache_only)

    def _run_query(self, query: Query) -> Response:
        assert not query.imgs, "TODO"
        client = openai.OpenAI()
        messages = [{"role": "user", "content": query.prompt, "type": "text"}]
        if query.hyperparameters is not None:
            kwargs = query.hyperparameters
        else:
            kwargs = {}
        completion = client.chat.completions.create(  # type: ignore[call-overload]
            messages=messages,
            model=self._model_name,
            **kwargs,
        )
        assert len(completion.choices) == 1
        text = completion.choices[0].message.content
        assert completion.usage is not None
        metadata = completion.usage.to_dict()
        return Response(text, metadata)


class CannedResponseModel(PretrainedLargeModel):
    """A model that returns responses from a dictionary and raises an error if
    no matching query is found.

    This is useful for development and testing.
    """

    def __init__(
        self,
        query_to_response: dict[Query, Response],
        cache_dir: Path,
        use_cache_only: bool = False,
    ) -> None:
        self._query_to_response = query_to_response
        super().__init__(cache_dir, use_cache_only)

    def get_id(self) -> str:
        return "canned"

    def _run_query(self, query: Query) -> Response:
        return self._query_to_response[query]
