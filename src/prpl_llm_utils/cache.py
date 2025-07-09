"""Methods for saving and loading model responses."""

import abc
import json
import logging
from pathlib import Path

from prpl_llm_utils.structs import Query, Response


class ResponseNotFound(Exception):
    """Raised during cache lookup if a response is not found."""


class PretrainedLargeModelCache(abc.ABC):
    """Base class for model caches."""

    @abc.abstractmethod
    def try_load_response(self, query: Query, model_id: str) -> Response:
        """Load a response or raise ResponseNotFound."""

    @abc.abstractmethod
    def save(self, query: Query, model_id: str, response: Response) -> None:
        """Save the response for the query."""


class FilePretrainedLargeModelCache(PretrainedLargeModelCache):
    """A cache that saves and loads from individual files."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(exist_ok=True)

    def _get_cache_dir_for_query(self, query: Query, model_id: str) -> Path:
        query_id = query.get_readable_id()
        cache_foldername = f"{model_id}_{query_id}"
        cache_folderpath = self._cache_dir / cache_foldername
        cache_folderpath.mkdir(exist_ok=True)
        return cache_folderpath

    def try_load_response(self, query: Query, model_id: str) -> Response:
        cache_dir = self._get_cache_dir_for_query(query, model_id)
        if not (cache_dir / "prompt.txt").exists():
            raise ResponseNotFound
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

    def save(self, query: Query, model_id: str, response: Response) -> None:
        cache_dir = self._get_cache_dir_for_query(query, model_id)
        # Cache the text prompt.
        prompt_file = cache_dir / "prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(query.prompt)
        # Cache the image prompt if it exists.
        if query.imgs is not None:
            imgs_folderpath = cache_dir / "imgs"
            imgs_folderpath.mkdir(exist_ok=True)
            for i, img in enumerate(query.imgs):
                filename_suffix = str(i) + ".jpg"
                img.save(imgs_folderpath / filename_suffix)
        # Cache the text response.
        completion_file = cache_dir / "completion.txt"
        with open(completion_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        # Cache the metadata.
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(response.metadata, f)
        logging.debug(f"Saved model response to {cache_dir}.")
