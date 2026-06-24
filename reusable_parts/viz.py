"""
viz.py
------
Plot the distribution of theses and grants over time, overlaid with method
emergence and reuse signals.

- Smooth line graph: one line for Thesis counts per year, one for Grant counts.
- Transparent vertical line: the year a canonical method first appears in the
  corpus (its "emergence" year).
- Dot on the line: a later thesis or grant (in a subsequent year) using that
  same canonical method ("reuse").

Requires a `canonical_name` column (from deduplication.py) to group methods.
Falls back to the raw `name` column if `canonical_name` isn't present yet —
in that case near-duplicate method names will be treated as distinct methods.

Usage
-----
    from viz import plot_thesis_grant_distribution

    fig, counts = plot_thesis_grant_distribution(full_df, save_path="distribution.png")
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)

COLORS = {"Thesis": "#4C72B0", "Grant": "#DD8452"}


def _classify_doc_type(academic_level: str) -> str:
    return "Grant" if str(academic_level).strip().lower() == "grant" else "Thesis"


def _document_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (source_title, year) -> counts of Thesis / Grant per year,
    reindexed to a continuous year range (gaps filled with 0).
    """
    docs = df.drop_duplicates(subset=["source_title", "year"]).copy()
    docs["doc_type"] = docs["academic_level"].apply(_classify_doc_type)

    year_range = range(int(docs["year"].min()), int(docs["year"].max()) + 1)
    counts = (
        docs.groupby(["year", "doc_type"]).size()
        .unstack(fill_value=0)
        .reindex(year_range, fill_value=0)
    )
    for col in ("Thesis", "Grant"):
        if col not in counts.columns:
            counts[col] = 0
    return counts[["Thesis", "Grant"]]


def _method_events(df: pd.DataFrame, method_col: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      first_year   : Series {method_col value -> first year it appears}
      reuse_points : DataFrame of (year, doc_type) rows for every occurrence
                     AFTER the method's first year — i.e. reuse events.
    """
    methods = df[df["category"] == "Category.method_technique"].copy()
    # Collapse so a thesis mentioning the same method twice counts once
    methods = methods.drop_duplicates(subset=["source_title", "year", method_col])
    methods["doc_type"] = methods["academic_level"].apply(_classify_doc_type)

    first_year = methods.groupby(method_col)["year"].min()
    methods["first_year"] = methods[method_col].map(first_year)
    reuse_points = methods[methods["year"] > methods["first_year"]]

    return first_year, reuse_points[["year", "doc_type", method_col]]


def plot_thesis_grant_distribution(
    df: pd.DataFrame,
    method_col: str = "canonical_name",
    save_path: Optional[str] = None,
    jitter: bool = True,
    jitter_seed: int = 42,
    figsize: Tuple[int, int] = (14, 7),
):
    """
    Build the smooth line graph + method emergence/reuse overlay.

    Parameters
    ----------
    df          : full requirements dataframe (one row per requirement,
                  needs columns: category, source_title, year, academic_level,
                  and method_col).
    method_col  : column identifying a canonical method. Use "canonical_name"
                  after running deduplication.py; falls back to "name" if
                  canonical_name isn't present.
    save_path   : if given, the figure is saved to this path (PNG).
    jitter      : add a small vertical jitter to overlapping reuse dots so
                  multiple reuses in the same year are visible as a cluster
                  rather than a single overlapping point. Purely visual —
                  does not affect the underlying counts.
    figsize     : matplotlib figure size.

    Returns
    -------
    (fig, counts) — the matplotlib Figure and the year x [Thesis, Grant]
    count DataFrame used to build it.
    """
    if method_col not in df.columns:
        logger.warning(
            "'%s' not found in df — falling back to 'name' (no deduplication "
            "applied, near-duplicate method names will be treated as distinct).",
            method_col,
        )
        method_col = "name"

    counts = _document_counts(df)
    first_year, reuse_points = _method_events(df, method_col)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(jitter_seed)

    # --- Smooth lines for Thesis / Grant counts ---
    for doc_type in ("Thesis", "Grant"):
        x = counts.index.values.astype(float)
        y = counts[doc_type].values.astype(float)

        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = np.clip(spline(x_smooth), 0, None)
        else:
            x_smooth, y_smooth = x, y

        ax.plot(
            x_smooth, y_smooth,
            label=doc_type, color=COLORS[doc_type], linewidth=2.5, zorder=3,
        )

    # --- Transparent vertical lines: method emergence years ---
    for yr in sorted(first_year.unique()):
        ax.axvline(yr, color="gray", alpha=0.08, linewidth=1.5, zorder=0)

    # --- Dots: method reuse in a later year, placed on the matching line ---
    for doc_type in ("Thesis", "Grant"):
        sub = reuse_points[reuse_points["doc_type"] == doc_type]
        if sub.empty:
            continue
        dot_years = sub["year"].values.astype(float)
        dot_y = counts.loc[sub["year"].values, doc_type].values.astype(float)

        if jitter:
            dot_y = dot_y + rng.uniform(-0.15, 0.15, size=len(dot_y)) * max(dot_y.max(), 1)

        ax.scatter(
            dot_years, dot_y,
            color=COLORS[doc_type], s=30, alpha=0.65, zorder=5,
            edgecolor="white", linewidth=0.6,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Count of Theses / Grants")
    ax.set_title("Thesis & Grant Volume Over Time\n(faint lines = new method emergence, dots = method reuse)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved figure → %s", save_path)

    return fig, counts
