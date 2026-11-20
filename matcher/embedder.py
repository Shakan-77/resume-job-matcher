"""Load a sentence-embedding model and turn text into vectors.

An *embedding* is a list of numbers (here 384 of them) that represents the
meaning of a piece of text. Texts with similar meaning get vectors pointing
in a similar direction, so we compare them with cosine similarity
(1.0 = same meaning, ~0 = unrelated).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np

from matcher import MODELS_DIR

# Small, fast models that fit easily next to other jobs on a 6 GB GPU.
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
FINETUNED_DIR = MODELS_DIR / "minilm-resume-jd"
# Use the fine-tuned model when it has been trained (python finetune.py), it scores best.
DEFAULT_MODEL = "finetuned" if FINETUNED_DIR.exists() else "minilm"


def resolve_model(name: str) -> str:
    """Map a short name ("minilm", "bge-small", "finetuned") or a path/hub id to something loadable."""
    if name == "finetuned":
        if not FINETUNED_DIR.exists():
            raise FileNotFoundError(f"No fine-tuned model at {FINETUNED_DIR}. Run finetune.py first.")
        return str(FINETUNED_DIR)
    return MODELS.get(name, name)


class Embedder:
    """Thin wrapper around SentenceTransformer with a per-text cache.

    Resumes and job descriptions repeat a lot of lines (and evaluation reuses
    the same documents many times), so caching saves a lot of GPU time.
    """

    def __init__(self, name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        import torch

        if device is None:
            # MATCHER_DEVICE=cpu forces CPU (useful when the GPU is busy with other jobs).
            device = os.environ.get("MATCHER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.batch_size = batch_size
        self.model = SentenceTransformer(resolve_model(name), device=device)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised embeddings (so dot product == cosine)."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            for text, vec in zip(missing, self._encode_with_retry(missing)):
                self._cache[text] = vec
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def _encode_with_retry(self, texts: list[str]) -> np.ndarray:
        """Encode, halving the batch size if the (shared) GPU runs out of memory."""
        import torch

        batch = self.batch_size
        while True:
            try:
                return self.model.encode(
                    texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                ).astype(np.float32)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    time.sleep(10)  # another process is using the GPU: wait and retry
                batch = max(1, batch // 2)
                self.batch_size = batch


@lru_cache(maxsize=4)
def get_embedder(name: str = DEFAULT_MODEL) -> Embedder:
    """Load each model only once per process."""
    return Embedder(name)
