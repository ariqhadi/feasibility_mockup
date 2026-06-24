"""
method_matcher.py
------------------
Given an input method/technique description, retrieve previously analyzed
thesis abstracts whose extracted requirements used the same or a similar
method.

This is a RETRIEVAL task, not a clustering task — there is no LLM involved.
Build an embedding index once over the existing requirements corpus, then
rank that corpus by cosine similarity to the query text. Reuses the same
embedding model as deduplication.py via embedding_utils.py so name+description
context is captured the same way.

Usage
-----
    from method_matcher import MethodMatcher

    matcher = MethodMatcher(df, name_col="name", desc_col="description")
    matches = matcher.search("logistic regression for binary classification",
                              top_k=10, extra_cols=["source_title", "year"])

    # Persist the index so you don't re-embed on every run:
    matcher.save_index("method_index.npz")
    matcher = MethodMatcher.load_index("method_index.npz", df)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from embedding_utils import embed_texts, DEFAULT_MODEL

logger = logging.getLogger(__name__)

DEFAULT_MIN_SIMILARITY = 0.45
# Rough calibration for all-MiniLM-L6-v2 cosine similarity on short technical text:
#   > 0.70   near-identical / same method
#   0.55-0.70  closely related method or sub-variant
#   0.40-0.55  loosely related (same broad family, e.g. both "regression")
#   < 0.40   unrelated


class MethodMatcher:
    """Embedding-based nearest-neighbour search over a requirements dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        name_col: str = "name",
        desc_col: str = "description",
        model_name: str = DEFAULT_MODEL,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.name_col = name_col
        self.desc_col = desc_col
        self.model_name = model_name

        if embeddings is not None:
            if len(embeddings) != len(self.df):
                raise ValueError(
                    f"Provided embeddings ({len(embeddings)} rows) don't match "
                    f"df ({len(self.df)} rows)."
                )
            self.embeddings = embeddings
        else:
            self.embeddings = self._build_index()

    def _build_index(self) -> np.ndarray:
        texts = (self.df[self.name_col] + ". " + self.df[self.desc_col]).tolist()
        logger.info("Building embedding index for %d rows…", len(texts))
        return embed_texts(texts, self.model_name)

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        extra_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return up to top_k rows most similar to *query*, sorted by similarity
        descending. Rows below min_similarity are dropped — if nothing clears
        the bar, an empty (but correctly shaped) DataFrame is returned.

        extra_cols: additional df columns to include in the output
                    (e.g. ["source_title", "year", "academic_level"]).
        """
        query_emb = embed_texts([query], self.model_name, show_progress_bar=False)[0]
        similarities = self.embeddings @ query_emb  # cosine similarity (both L2-normalised)

        order = np.argsort(-similarities)
        results = self.df.iloc[order].copy()
        results["similarity"] = similarities[order]
        results = results[results["similarity"] >= min_similarity].head(top_k)

        cols = [self.name_col, self.desc_col, "similarity"]
        if extra_cols:
            cols += [c for c in extra_cols if c in results.columns and c not in cols]

        return results[cols].reset_index(drop=True)

    def search_many(
        self,
        queries: List[str],
        top_k: int = 10,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        extra_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Batch version of search(). Returns one combined DataFrame with a
        leading 'query' column identifying which input each row matched.
        More efficient than calling search() in a loop — embeds all queries
        in a single batch.
        """
        query_embs = embed_texts(queries, self.model_name, show_progress_bar=False)
        all_rows = []

        for query, query_emb in zip(queries, query_embs):
            similarities = self.embeddings @ query_emb
            order = np.argsort(-similarities)
            results = self.df.iloc[order].copy()
            results["similarity"] = similarities[order]
            results = results[results["similarity"] >= min_similarity].head(top_k)
            results.insert(0, "query", query)
            all_rows.append(results)

        cols = ["query", self.name_col, self.desc_col, "similarity"]
        if extra_cols:
            cols += [c for c in extra_cols if c in self.df.columns and c not in cols]

        if not all_rows:
            return pd.DataFrame(columns=cols)

        combined = pd.concat(all_rows, ignore_index=True)
        return combined[cols]

    def save_index(self, path: str) -> None:
        """Persist embeddings so the index can be reloaded without re-embedding."""
        np.savez(path, embeddings=self.embeddings)
        logger.info("Saved index (%d rows) → %s", len(self.df), path)

    @classmethod
    def load_index(
        cls,
        path: str,
        df: pd.DataFrame,
        name_col: str = "name",
        desc_col: str = "description",
        model_name: str = DEFAULT_MODEL,
    ) -> "MethodMatcher":
        """
        Reload a saved index. df must be the SAME dataframe (same rows, same
        order) that was used to build the saved index — there is no row-id
        reconciliation, so a mismatched df will silently misalign results.
        """
        data = np.load(path)
        embeddings = data["embeddings"]
        if len(embeddings) != len(df):
            raise ValueError(
                f"Index has {len(embeddings)} rows but df has {len(df)} rows — "
                "the index was likely built from a different dataframe."
            )
        return cls(
            df, name_col=name_col, desc_col=desc_col, model_name=model_name, embeddings=embeddings
        )
