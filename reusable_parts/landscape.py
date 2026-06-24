"""
landscape.py
------------
Turn the year-by-year historical-analysis output (year_summaries.json) into a
chart-friendly shape, and render it as an interactive Plotly figure with
turning points overlaid.

Expected input shapes
----------------------
year_summaries : dict, keyed by year string, e.g.

    {
      "2017": {
        "year": "2017",
        "summary": "...",
        "requirements": [
          {"category": "data", "name": "...", "description": "...", "search_query": "..."},
          ...
        ]
      },
      ...
    }

all_year_summaries : dict with keys
    "narrative"                 : str
    "turning_points"            : [{"period": "2021", "change": "...", "evidence": "..."}, ...]
    "persistent_requirements"   : [{"name": "...", "note": "..."}, ...]
    "year_specific_requirements": [{"name": "...", "year": "...", "note": "..."}, ...]
    "unexplained_patterns"      : [str, ...]
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

CATEGORY_COLORS = {
    "data": "#4C72B0",
    "method_technique": "#DD8452",
    "tool_library": "#55A868",
    "compute": "#C44E52",
    "human_effort": "#8172B2",
    "other": "#937860",
}

CATEGORY_LABELS = {
    "data": "Data",
    "method_technique": "Method / Technique",
    "tool_library": "Tool / Library",
    "compute": "Compute",
    "human_effort": "Human Effort",
    "other": "Other",
}


def requirement_counts_by_year(year_summaries: Dict[str, dict]) -> pd.DataFrame:
    """
    Long-format dataframe: one row per (year, category) with the count of
    requirements in that category for that year. Years with zero requirements
    in a category are NOT included (Plotly handles the gaps fine for a
    stacked area chart via fillna(0) on pivot).
    """
    rows = []
    for year_str, payload in year_summaries.items():
        try:
            year = int(year_str)
        except ValueError:
            continue
        for req in payload.get("requirements", []):
            rows.append({"year": year, "category": req.get("category", "other")})

    if not rows:
        return pd.DataFrame(columns=["year", "category", "count"])

    df = pd.DataFrame(rows)
    counts = df.groupby(["year", "category"]).size().reset_index(name="count")
    return counts.sort_values("year")


def plot_landscape(
    year_summaries: Dict[str, dict],
    turning_points: List[dict] | None = None,
    height: int = 500,
    width: int = 1000,
):
    """
    Stacked-area Plotly chart of requirement-category counts per year, with
    a translucent vertical line + hover label at each turning point's year.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    counts = requirement_counts_by_year(year_summaries)
    if counts.empty:
        return go.Figure()

    pivot = counts.pivot(index="year", columns="category", values="count").fillna(0)
    all_years = range(int(pivot.index.min()), int(pivot.index.max()) + 1)
    pivot = pivot.reindex(all_years, fill_value=0)

    fig = go.Figure()

    # Stable stacking order so the chart doesn't reshuffle colors/order
    ordered_categories = [c for c in CATEGORY_LABELS if c in pivot.columns] + [
        c for c in pivot.columns if c not in CATEGORY_LABELS
    ]

    for category in ordered_categories:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[category],
            mode="lines",
            stackgroup="categories",
            name=CATEGORY_LABELS.get(category, category),
            line=dict(width=0.5, color=CATEGORY_COLORS.get(category, "#888888")),
            fillcolor=CATEGORY_COLORS.get(category, "#888888"),
            hovertemplate=(
                f"{CATEGORY_LABELS.get(category, category)}<br>"
                "Year %{x}<br>Count %{y}<extra></extra>"
            ),
        ))

    y_max = float(pivot.sum(axis=1).max()) if len(pivot) else 1.0

    for tp in (turning_points or []):
        try:
            year = int(tp["period"])
        except (KeyError, ValueError, TypeError):
            continue
        if year not in pivot.index:
            continue
        fig.add_trace(go.Scatter(
            x=[year] * 15,
            y=[i * (y_max * 1.08 / 14) for i in range(15)],
            mode="lines",
            line=dict(color="black", width=3, dash="dot"),
            opacity=0.35,
            showlegend=False,
            hovertemplate=f"<b>Turning point ({year})</b><br>{tp.get('change', '')}<extra></extra>",
        ))

    fig.update_layout(
        title="Requirement Landscape by Year",
        xaxis_title="Year",
        yaxis_title="Number of requirements",
        height=height,
        width=width,
        hovermode="closest",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=80),
    )
    return fig
