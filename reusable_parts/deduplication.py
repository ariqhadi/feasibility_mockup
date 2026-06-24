"""
deduplication.py
----------------
Robust deduplication of requirement names extracted from thesis abstracts.

Pipeline — one pass, no feedback loops:

  Phase 1 | Embed
    Encode (name + description) with a sentence-transformer.
    Descriptions carry the contextual signal that name similarity alone misses
    (e.g. "logistic regression for binary outcome" vs "regression as general
    statistical approach").

  Phase 2 | Cluster
    Agglomerative clustering with cosine distance at a tight threshold.
    Errs toward more / smaller clusters — it is safer to over-split here
    because Phase 4 can still merge clusters whose canonical names are
    near-identical, but there is no recovery from an over-merge.

  Phase 3 | Name
    For each cluster the LLM receives the full {name, description} of every
    member and returns ONE canonical name.  The LLM never decides structure —
    only names what is already in front of it.  This eliminates the
    merge/split feedback loop.

  Phase 4 | Fuzzy dedup
    rapidfuzz token_sort_ratio on the canonical names produced in Phase 3.
    Two canonical names above the similarity threshold are merged via
    union-find into one final cluster ID.  Fully deterministic — no LLM.

Dependencies
------------
    pip install sentence-transformers scikit-learn rapidfuzz

Usage
-----
    from deduplication import deduplicate_requirements

    result_df = deduplicate_requirements(df, llm)
    # Adds columns: cluster_label (int), canonical_name (str)
"""

from __future__ import annotations

import json
import logging
import re
import time
import random
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering

from embedding_utils import embed_texts, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DISTANCE_THRESHOLD = 0.28     # cosine distance; tune on a sample
                                      # lower = tighter clusters (more of them)
                                      # 0.28 ≈ cosine similarity 0.72
DEFAULT_FUZZY_THRESHOLD = 88          # rapidfuzz score 0-100; above = same cluster


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

NAMING_PROMPT = """\
You are a research taxonomy assistant.

Below is a group of requirement names from academic thesis abstracts that have \
been identified as semantically related. Assign ONE canonical name that best \
represents the whole group.

Rules:
- 3–6 words, Title Case
- Prefer the most general label that still distinguishes this group from other
  techniques (e.g. "Sentiment Analysis" not "Text Analysis"; "Logistic
  Regression" not "Regression")
- Do NOT merge or split — treat all items below as one group

Members:
{members}

Return ONLY a JSON object and nothing else:
{{"canonical_name": "<string>"}}
"""


# ---------------------------------------------------------------------------
# Phase 1 — Embed
# ---------------------------------------------------------------------------

def _embed(df: pd.DataFrame, name_col: str, desc_col: str, model_name: str) -> np.ndarray:
    """Return L2-normalised embeddings for (name + '. ' + description)."""
    texts = (df[name_col] + ". " + df[desc_col]).tolist()
    logger.info("Embedding %d rows…", len(texts))
    return embed_texts(texts, model_name)


# ---------------------------------------------------------------------------
# Phase 2 — Cluster
# ---------------------------------------------------------------------------

def _cluster(embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
    """
    Agglomerative clustering with cosine distance.
    Returns an array of integer cluster labels aligned with embeddings rows.
    """
    logger.info(
        "Clustering %d embeddings (cosine distance threshold=%.3f)…",
        len(embeddings),
        distance_threshold,
    )
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels))
    logger.info("Found %d clusters (avg size %.1f)", n_clusters, len(labels) / n_clusters)
    return labels


# ---------------------------------------------------------------------------
# Phase 3 — LLM names each cluster
# ---------------------------------------------------------------------------

def _extract_canonical(text: str) -> str:
    """Pull canonical_name from LLM JSON response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    obj = json.loads(text[start : end + 1])
    return obj["canonical_name"]


def _safe_invoke(llm: Any, prompt: str, retries: int = 3, backoff: float = 4.0) -> str:
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (attempt + 1) + random.uniform(0, 2)
            logger.warning("LLM call failed (%s), retrying in %.1fs…", exc, wait)
            time.sleep(wait)


def _name_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    name_col: str,
    desc_col: str,
    llm: Any,
    delay: tuple[int, int],
) -> dict[int, str]:
    """
    Return {cluster_label -> canonical_name}.
    Single-member clusters skip the LLM entirely (the name is already unique).
    """
    cluster_to_name: dict[int, str] = {}
    unique_labels = sorted(set(labels))
    total = len(unique_labels)

    for i, label in enumerate(unique_labels, 1):
        mask = labels == label
        members_df = df[mask]

        # Single-member cluster: no LLM needed
        if len(members_df) == 1:
            cluster_to_name[label] = members_df.iloc[0][name_col]
            continue

        members_text = "\n".join(
            f'- "{row[name_col]}": {row[desc_col]}'
            for _, row in members_df.iterrows()
        )
        prompt = NAMING_PROMPT.format(members=members_text)

        logger.info("Naming cluster %d/%d (%d members)…", i, total, len(members_df))
        try:
            raw = _safe_invoke(llm, prompt)
            cluster_to_name[label] = _extract_canonical(raw)
        except Exception as exc:
            logger.error("Cluster %d naming failed (%s) — using most common name.", label, exc)
            cluster_to_name[label] = members_df[name_col].mode().iloc[0]

        if i < total:
            time.sleep(random.randint(*delay))

    return cluster_to_name


# ---------------------------------------------------------------------------
# Phase 4 — Fuzzy dedup on canonical names
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def groups(self) -> dict:
        result: dict = {}
        for k in self.parent:
            root = self.find(k)
            result.setdefault(root, []).append(k)
        return result


def _fuzzy_merge_canonical(
    cluster_to_name: dict[int, str], threshold: int
) -> dict[int, tuple[int, str]]:
    """
    Merge clusters whose canonical names are near-identical (fuzzy score >= threshold).
    Returns {original_cluster_label -> (final_cluster_id, final_canonical_name)}.
    Uses the shortest canonical name as the representative for merged groups,
    which tends to be the more general label.
    """
    labels = list(cluster_to_name.keys())
    names = [cluster_to_name[l] for l in labels]

    uf = _UnionFind(labels)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = fuzz.token_sort_ratio(names[i], names[j])
            if score >= threshold:
                logger.debug(
                    "Merging '%s' + '%s' (score=%d)", names[i], names[j], score
                )
                uf.union(labels[i], labels[j])

    groups = uf.groups()
    n_merged = len(labels) - len(groups)
    if n_merged:
        logger.info("Fuzzy dedup merged %d clusters → %d final clusters", n_merged, len(groups))

    # Build mapping: original label → (final_id, canonical_name)
    # canonical name = shortest name in the group (tends to be most general)
    mapping: dict[int, tuple[int, str]] = {}
    for final_id, (root, members) in enumerate(groups.items(), 1):
        best_name = min((cluster_to_name[m] for m in members), key=len)
        for m in members:
            mapping[m] = (final_id, best_name)

    return mapping


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def deduplicate_requirements(
    df: pd.DataFrame,
    llm: Any,
    name_col: str = "name",
    desc_col: str = "description",
    embedding_model: str = DEFAULT_MODEL,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD,
    delay: tuple[int, int] = (1, 3),
) -> pd.DataFrame:
    """
    Add `canonical_name` and `cluster_id` columns to a copy of *df*.

    Parameters
    ----------
    df                 : DataFrame with requirement names and descriptions.
    llm                : LangChain-compatible chat model (.invoke()).
    name_col           : Column with raw requirement name.
    desc_col           : Column with one-sentence description.
    embedding_model    : sentence-transformers model name.
    distance_threshold : Cosine distance cutoff for agglomerative clustering.
                         Lower = tighter clusters (more of them).
                         Recommended range: 0.20 – 0.35.
                         Run explore_threshold() first to pick a good value.
    fuzzy_threshold    : rapidfuzz token_sort_ratio cutoff for merging
                         near-identical canonical names (0–100).
    delay              : (min, max) seconds between LLM calls.

    Returns
    -------
    Copy of df with two new columns: canonical_name (str), cluster_id (int).
    """
    result = df.copy().reset_index(drop=True)

    # Phase 1 — embed
    embeddings = _embed(result, name_col, desc_col, embedding_model)

    # Phase 2 — cluster
    labels = _cluster(embeddings, distance_threshold)

    # Phase 3 — LLM names each cluster
    logger.info("Phase 3: naming %d clusters with LLM…", len(set(labels)))
    cluster_to_name = _name_clusters(result, labels, name_col, desc_col, llm, delay)

    # Phase 4 — fuzzy dedup on canonical names
    logger.info("Phase 4: fuzzy dedup on canonical names (threshold=%d)…", fuzzy_threshold)
    mapping = _fuzzy_merge_canonical(cluster_to_name, fuzzy_threshold)

    result["cluster_id"] = [mapping[l][0] for l in labels]
    result["canonical_name"] = [mapping[l][1] for l in labels]

    logger.info(
        "Done. %d rows → %d final clusters.",
        len(result),
        result["cluster_id"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Utility: explore threshold on a sample
# ---------------------------------------------------------------------------

def explore_threshold(
    df: pd.DataFrame,
    name_col: str = "name",
    desc_col: str = "description",
    embedding_model: str = DEFAULT_MODEL,
    thresholds: list[float] | None = None,
    sample_n: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Try several distance thresholds on a sample and report cluster statistics.
    Use this to pick a good distance_threshold before running the full pipeline.

    Returns a DataFrame with columns:
        threshold, n_clusters, avg_size, max_size, singleton_pct
    """
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.28, 0.32, 0.38, 0.45]

    sample = df.sample(min(sample_n, len(df)), random_state=random_state)
    embeddings = _embed(sample, name_col, desc_col, embedding_model)

    rows = []
    for t in thresholds:
        labels = _cluster(embeddings, t)
        unique, counts = np.unique(labels, return_counts=True)
        rows.append({
            "threshold": t,
            "n_clusters": len(unique),
            "avg_size": round(counts.mean(), 1),
            "max_size": int(counts.max()),
            "singleton_pct": round((counts == 1).mean() * 100, 1),
        })

    return pd.DataFrame(rows)
