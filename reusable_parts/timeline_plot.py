"""
timeline_plot.py
-----------------
Plot the yearly distribution of theses and grants as two smooth lines, then
overlay each extracted method as:
  - a translucent vertical line at the year the method first appeared
    (its "origin year", e.g. from method_history.get_origin_year)
  - a dot sitting on the relevant curve (thesis or grant) at every year
    where a similar method was found in the corpus via MethodMatcher

Two renderers are provided:
  - plot_method_adoption_timeline()         -> matplotlib, static
  - plot_method_adoption_timeline_plotly()  -> Plotly, interactive
    (hover tooltips on lines/dots, zoom, pan, legend click to toggle a
    method's vertical line + dots together)

Usage
-----
    from timeline_plot import plot_method_adoption_timeline_plotly

    methods = [
        {"name": "Word2vec", "search_query": "Word2vec text analysis", "origin_year": 2013},
        {"name": "Sentiment Analysis", "search_query": "Sentiment analysis finance texts", "origin_year": 2002},
        {"name": "Logistic Regression", "search_query": "logistic regression classification", "origin_year": 1958},
    ]

    fig = plot_method_adoption_timeline_plotly(
        abstracts_df,            # one row per thesis/grant
        methods,
        matcher,                 # MethodMatcher built over the per-requirement data
        year_col="Published Year",
        level_col="Academic Level",
        matched_year_col="year",
        matched_level_col="academic_level",
    )
    fig.show()
"""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from method_matcher import MethodMatcher

logger = logging.getLogger(__name__)


def _smooth_curve(years: List[int], counts: List[float], points_per_year: int = 20):
    """
    Interpolate a smooth curve through (years, counts) using a cubic spline.
    Falls back to the raw points if there aren't enough to fit a spline.
    Clips negative dips introduced by spline overshoot back to 0, since
    counts can't be negative.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(counts, dtype=float)

    if len(x) < 4:
        return x, y

    x_smooth = np.linspace(x.min(), x.max(), max(len(x) * points_per_year, 2))
    spline = make_interp_spline(x, y, k=3)
    y_smooth = np.clip(spline(x_smooth), 0, None)
    return x_smooth, y_smooth


def plot_method_adoption_timeline(
    abstracts_df: pd.DataFrame,
    methods: List[dict],
    matcher: MethodMatcher,
    year_col: str = "Published Year",
    level_col: str = "Academic Level",
    grant_value: str = "grant",
    matched_year_col: Optional[str] = None,
    matched_level_col: Optional[str] = None,
    min_similarity: float = 0.45,
    top_k: int = 50,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Parameters
    ----------
    abstracts_df : one row per thesis/grant — used to compute the base
                   distribution curves. Must contain year_col and level_col.
    methods      : list of dicts, each with:
                     - "name": display name (used in the legend)
                     - "search_query": query used against the matcher, and
                       optionally for the origin-year lookup
                     - "origin_year": int year the method first appeared,
                       or None to skip drawing the vertical line
    matcher      : a MethodMatcher built over the PER-REQUIREMENT dataframe
                   (one row per extracted requirement, many per abstract).
                   Its underlying df must expose the source abstract's year
                   and academic level via matched_year_col / matched_level_col.
    year_col, level_col : column names in abstracts_df.
    matched_year_col, matched_level_col : column names in matcher.df.
                   Default to year_col / level_col if not given (covers the
                   case where both dataframes use the same column names).
    min_similarity : similarity cutoff passed to matcher.search().
    top_k        : max matches considered per method.
    save_path    : if given, the figure is also saved to this path.

    Returns
    -------
    (fig, ax)
    """
    matched_year_col = matched_year_col or year_col
    matched_level_col = matched_level_col or level_col

    work = abstracts_df.dropna(subset=[year_col]).copy()
    work[year_col] = work[year_col].astype(int)

    all_years = range(int(work[year_col].min()), int(work[year_col].max()) + 1)
    is_grant = work[level_col] == grant_value

    thesis_counts = work[~is_grant].groupby(year_col).size().reindex(all_years, fill_value=0)
    grant_counts = work[is_grant].groupby(year_col).size().reindex(all_years, fill_value=0)

    x_thesis, y_thesis = _smooth_curve(thesis_counts.index.tolist(), thesis_counts.values.tolist())
    x_grant, y_grant = _smooth_curve(grant_counts.index.tolist(), grant_counts.values.tolist())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_thesis, y_thesis, label="Theses", color="#3b6ea5", linewidth=2.5)
    ax.plot(x_grant, y_grant, label="Grants", color="#c0533e", linewidth=2.5)

    color_cycle = plt.cm.tab10.colors

    for i, method in enumerate(methods):
        color = color_cycle[i % len(color_cycle)]
        origin_year = method.get("origin_year")

        if origin_year is not None:
            ax.axvline(
                origin_year,
                color=color,
                alpha=0.25,
                linewidth=4,
                label=f"{method['name']} (origin {origin_year})",
                zorder=1,
            )

        query = method.get("search_query", method["name"])
        matches = matcher.search(
            query,
            top_k=top_k,
            min_similarity=min_similarity,
            extra_cols=[matched_year_col, matched_level_col],
        )

        if matches.empty:
            logger.info("No corpus matches found for method '%s'.", method["name"])
            continue

        for _, row in matches.iterrows():
            yr = row[matched_year_col]
            if pd.isna(yr):
                continue
            yr = int(yr)
            is_g = row[matched_level_col] == grant_value
            curve_x, curve_y = (x_grant, y_grant) if is_g else (x_thesis, y_thesis)
            if len(curve_x) == 0:
                continue
            y_val = np.interp(yr, curve_x, curve_y)
            ax.scatter(
                yr, y_val,
                color=color, edgecolor="black", linewidth=0.6,
                zorder=5, s=70,
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_title("Thesis / Grant Distribution with Method Adoption Timeline")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.margins(x=0.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)

    return fig, ax


# ---------------------------------------------------------------------------
# Plotly (interactive) renderer
# ---------------------------------------------------------------------------

_PLOTLY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_method_adoption_timeline_plotly(
    abstracts_df: pd.DataFrame,
    methods: List[dict],
    matcher: MethodMatcher,
    year_col: str = "Published Year",
    level_col: str = "Academic Level",
    grant_value: str = "grant",
    matched_year_col: Optional[str] = None,
    matched_level_col: Optional[str] = None,
    matched_title_col: Optional[str] = None,
    input_year: Optional[int] = None,
    min_similarity: float = 0.45,
    top_k: int = 50,
    height: int = 600,
    width: int = 1000,
    save_path: Optional[str] = None,
):
    """
    Interactive Plotly version of plot_method_adoption_timeline().

    Same parameters as the matplotlib version, plus:
      matched_title_col : optional column in matcher.df (e.g. source title)
                          shown in the hover tooltip for matched dots.
      input_year        : the published/submission year of the INPUT abstract
                          itself (the one input_methods were extracted from).
                          Drawn as a black star marker on the x-axis so you
                          can see where the input sits relative to the corpus
                          and the methods' origin years. None to skip.
      height, width     : figure size in pixels.
      save_path         : if given and ends in .html, saves a standalone
                          interactive HTML file; otherwise saves a static
                          image (requires kaleido).

    Methods are drawn in ascending order of origin_year (methods with no
    origin_year are placed last), so the legend reads chronologically.

    Interactivity:
      - Hover over either base line to see the year/count.
      - Hover anywhere along a method's translucent vertical line to see its
        name and origin year.
      - Hover over a dot to see the method, year, similarity score, academic
        level, and (if provided) source title.
      - Hover over the input-year star to confirm the input abstract's year.
      - Click a legend entry to toggle that method's line + dots together
        (they share a legendgroup).
      - Scroll/drag to zoom, double-click to reset.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    matched_year_col = matched_year_col or year_col
    matched_level_col = matched_level_col or level_col

    # Chronological legend order: methods with a known origin_year first
    # (ascending), unknowns last, preserving their relative input order.
    methods = sorted(
        methods,
        key=lambda m: (m.get("origin_year") is None, m.get("origin_year") or 0),
    )

    work = abstracts_df.dropna(subset=[year_col]).copy()
    work[year_col] = work[year_col].astype(int)

    all_years = range(int(work[year_col].min()), int(work[year_col].max()) + 1)
    is_grant = work[level_col] == grant_value

    thesis_counts = work[~is_grant].groupby(year_col).size().reindex(all_years, fill_value=0)
    grant_counts = work[is_grant].groupby(year_col).size().reindex(all_years, fill_value=0)

    x_thesis, y_thesis = _smooth_curve(thesis_counts.index.tolist(), thesis_counts.values.tolist())
    x_grant, y_grant = _smooth_curve(grant_counts.index.tolist(), grant_counts.values.tolist())

    # Fix the visible x-range to the corpus's own year span (+ a small margin).
    # Without this, a method whose origin year is decades before the corpus
    # starts (e.g. corpus begins 2000, a method originates 1923) would force
    # Plotly to autorange all the way back to 1923, squashing the actual
    # thesis/grant curves into a sliver. Origin years outside this range are
    # instead clamped to the nearest edge and labelled with their true year.
    data_x_min = int(work[year_col].min())
    data_x_max = int(work[year_col].max())
    margin = max(1, round((data_x_max - data_x_min) * 0.03))
    view_x_min = data_x_min - margin
    view_x_max = data_x_max + margin

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_thesis, y=y_thesis, mode="lines", name="Theses",
        line=dict(color="#3b6ea5", width=3),
        hovertemplate="Theses<br>Year %{x:.0f}<br>Count %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_grant, y=y_grant, mode="lines", name="Grants",
        line=dict(color="#c0533e", width=3),
        hovertemplate="Grants<br>Year %{x:.0f}<br>Count %{y:.1f}<extra></extra>",
    ))

    y_max = max(
        float(np.max(y_thesis)) if len(y_thesis) else 0.0,
        float(np.max(y_grant)) if len(y_grant) else 0.0,
        1.0,
    )

    annotations = []

    if input_year is not None:
        clamped_input = max(view_x_min, min(input_year, view_x_max))
        is_input_clamped = clamped_input != input_year
        fig.add_trace(go.Scatter(
            x=[clamped_input], y=[-y_max * 0.04],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="black"),
            text=["Input"],
            textposition="bottom center",
            name=f"Input abstract ({input_year})",
            hovertemplate=(
                f"Input abstract<br>Year: {input_year}"
                + (" (off-chart, clamped to edge)" if is_input_clamped else "")
                + "<extra></extra>"
            ),
        ))

    for i, method in enumerate(methods):
        color = _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        name = method["name"]
        legend_group = f"method-{i}"
        origin_year = method.get("origin_year")

        if origin_year is not None:
            clamped = max(data_x_min, min(origin_year, data_x_max))
            is_clamped = clamped != origin_year

            # Drawn as a Scatter line (not a layout shape) so it's hoverable
            # along its whole length, not just at two endpoints.
            n_pts = 25
            fig.add_trace(go.Scatter(
                x=[clamped] * n_pts,
                y=np.linspace(0, y_max * 1.05, n_pts),
                mode="lines",
                line=dict(color=color, width=10, dash="dash" if is_clamped else "solid"),
                opacity=0.25,
                name=f"{name} (origin {origin_year})",
                legendgroup=legend_group,
                hovertemplate=(
                    f"{name}<br>Origin year: {origin_year}"
                    + (" (off-chart, clamped to edge)" if is_clamped else "")
                    + "<extra></extra>"
                ),
            ))

            if is_clamped:
                arrow = "◄" if origin_year < data_x_min else "►"
                annotations.append(dict(
                    x=clamped, y=y_max * 1.05,
                    text=f"{arrow} {name}: {origin_year}",
                    showarrow=False,
                    font=dict(color=color, size=11),
                    xanchor="left" if origin_year < data_x_min else "right",
                    yanchor="bottom",
                ))

        query = method.get("search_query", name)
        extra_cols = [matched_year_col, matched_level_col]
        if matched_title_col:
            extra_cols.append(matched_title_col)

        matches = matcher.search(
            query,
            top_k=top_k,
            min_similarity=min_similarity,
            extra_cols=extra_cols,
        )

        if matches.empty:
            logger.info("No corpus matches found for method '%s'.", name)
            continue

        dot_x, dot_y, dot_text = [], [], []
        for _, row in matches.iterrows():
            yr = row[matched_year_col]
            if pd.isna(yr):
                continue
            yr = int(yr)
            is_g = row[matched_level_col] == grant_value
            curve_x, curve_y = (x_grant, y_grant) if is_g else (x_thesis, y_thesis)
            if len(curve_x) == 0:
                continue
            y_val = float(np.interp(yr, curve_x, curve_y))
            dot_x.append(yr)
            dot_y.append(y_val)

            sim = row.get("similarity")
            title = row.get(matched_title_col) if matched_title_col else None
            label_lines = [
                f"<b>{name}</b>",
                f"Year: {yr}",
                f"Type: {'Grant' if is_g else 'Thesis'}",
            ]
            if sim is not None:
                label_lines.append(f"Similarity: {sim:.2f}")
            if title:
                label_lines.append(f"Source: {title}")
            dot_text.append("<br>".join(label_lines))

        if not dot_x:
            continue

        fig.add_trace(go.Scatter(
            x=dot_x, y=dot_y, mode="markers",
            marker=dict(color=color, size=11, line=dict(color="black", width=1)),
            name=name,
            legendgroup=legend_group,
            showlegend=origin_year is None,  # avoid duplicate legend entry
            text=dot_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Thesis / Grant Distribution with Method Adoption Timeline",
        xaxis_title="Year",
        yaxis_title="Count",
        xaxis=dict(range=[view_x_min, view_x_max]),
        height=height,
        width=width,
        hovermode="closest",
        legend=dict(
            groupclick="togglegroup",
            x=1.02, y=1, xanchor="left", yanchor="top",
        ),
        margin=dict(r=220, t=80, b=60),
        annotations=annotations,
    )

    if save_path:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        logger.info("Saved figure → %s", save_path)

    return fig
