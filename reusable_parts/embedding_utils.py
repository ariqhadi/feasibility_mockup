"""
embedding_utils.py
-------------------
Shared sentence-embedding helpers used by both deduplication.py and
method_matcher.py. The model loader is cached so repeated calls within the
same session don't reload the model from disk.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"   # fast, CPU-friendly, ~80 MB


@lru_cache(maxsize=4)
def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading sentence-transformer '%s'…", model_name)
    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Return L2-normalised embeddings (so dot product == cosine similarity)."""
    model = get_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress_bar)
