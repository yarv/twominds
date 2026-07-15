"""Pluggable embedding backends for clustering responses.

Backends (selectable, comparable side by side):
- ``local``          : sentence-transformers on CPU/GPU (default BAAI/bge-small-en-v1.5)
- ``openai-3-small`` : OpenAI text-embedding-3-small
- ``openai-3-large`` : OpenAI text-embedding-3-large

All backends return L2-normalised float32 vectors, so cosine similarity is a dot
product and cosine distance is ``1 - dot``.
"""

from __future__ import annotations

import sys
from typing import Protocol

import numpy as np

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
_OPENAI_MODELS = {
    "openai-3-small": "text-embedding-3-small",
    "openai-3-large": "text-embedding-3-large",
}
BACKENDS = ["local", "openai-3-small", "openai-3-large"]


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:  # (len(texts), dim), L2-normalised
        ...


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        self.name = f"local:{model_name}"
        self._model_name = model_name
        self._model = None  # lazy load (avoids the download/import unless used)

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "the 'local' embedding backend needs sentence-transformers "
                    "(+ torch), which are not part of the default install. "
                    "Opt in with `uv sync --group local-embeddings`, or use the "
                    "default API backend: -b openai-3-small (needs "
                    "OPENAI_API_KEY)."
                ) from e

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        # progress bar on interactive terminals only (tqdm writes to stderr);
        # local embedding of a big run takes long enough to look hung without it
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=sys.stderr.isatty(),
        )
        return _l2_normalise(np.asarray(vecs, dtype=np.float32))


class OpenAIEmbedder:
    def __init__(self, backend: str):
        if backend not in _OPENAI_MODELS:
            raise ValueError(f"unknown OpenAI embedding backend: {backend}")
        self.name = backend
        self._model = _OPENAI_MODELS[backend]
        self._client = None

    def _ensure(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            self._client = OpenAI()

    _BATCH = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        # OpenAI rejects empty strings; substitute a single space.
        cleaned = [t if t.strip() else " " for t in texts]
        vecs: list[list[float]] = []
        for i in range(0, len(cleaned), self._BATCH):
            chunk = cleaned[i : i + self._BATCH]
            resp = self._client.embeddings.create(model=self._model, input=chunk)
            vecs.extend(d.embedding for d in resp.data)
        return _l2_normalise(np.array(vecs, dtype=np.float32))


def get_embedder(backend: str, *, local_model: str = DEFAULT_LOCAL_MODEL) -> Embedder:
    if backend == "local":
        return LocalEmbedder(local_model)
    if backend in _OPENAI_MODELS:
        return OpenAIEmbedder(backend)
    raise ValueError(f"unknown embedding backend '{backend}' (choices: {BACKENDS})")
