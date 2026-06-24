"""
app.py
------
Streamlit dashboard for the thesis-feasibility historical analysis.

Layout
------
1. Year-over-year requirement landscape (stacked area chart) with turning
   points overlaid — precomputed, shown immediately on load.
2. Turning points, persistent requirements, year-specific requirements, and
   the overall narrative — also precomputed, shown immediately on load.
3. Per-year drill-down (pick a year, see its summary + requirement table).
4. "Check feasibility" box: paste an abstract, optionally its year, and hit
   Analyze. This is the only part that reprocesses — everything above is
   loaded once and cached. On Analyze it: extracts methods from the abstract
   with the LLM, looks up each method's origin year via Semantic Scholar,
   matches against the existing requirements corpus via MethodMatcher, and
   plots an interactive adoption-timeline chart.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from boilerplate import AnalysisResult, Category, prompt
from landscape import plot_landscape
from method_matcher import MethodMatcher
from method_history import get_origin_years
from timeline_plot import plot_method_adoption_timeline_plotly

# ---------------------------------------------------------------------------
# Paths — adjust here if your data lives elsewhere.
# ---------------------------------------------------------------------------

YEAR_SUMMARIES_PATH = "year_summaries.json"
ALL_YEAR_SUMMARIES_PATH = "all_year_summaries.json"
REQUIREMENTS_XLSX_PATH = "abstract_requirements 2.xlsx"
ABSTRACTS_XLSX_PATH = "/Users/ariq/Public/Data/thesis_feasibility/New Dataset.xlsx"
ABSTRACTS_SHEET = "filtered"

st.set_page_config(page_title="Thesis Feasibility — Historical Analysis", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders — run once per server process, not on every interaction.
# ---------------------------------------------------------------------------

@st.cache_data
def load_year_summaries() -> dict:
    with open(YEAR_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_all_year_summaries() -> dict:
    with open(ALL_YEAR_SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_method_df() -> pd.DataFrame:
    df = pd.read_excel(REQUIREMENTS_XLSX_PATH)
    return df[df["category"] == "Category.method_technique"].copy().reset_index(drop=True)


@st.cache_data
def load_abstracts_df() -> pd.DataFrame:
    return pd.read_excel(ABSTRACTS_XLSX_PATH, sheet_name=ABSTRACTS_SHEET)


@st.cache_resource
def load_llm():
    from tools import get_model
    return get_model()


@st.cache_resource
def build_matcher(_method_df: pd.DataFrame) -> MethodMatcher:
    # _method_df is excluded from the cache key (leading underscore) since a
    # DataFrame isn't hashable — the underlying xlsx file is the real cache key.
    return MethodMatcher(_method_df, name_col="name", desc_col="description")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Thesis Feasibility — Historical Analysis")
st.caption(
    "Year-over-year view of what past theses/grants required, plus a tool to "
    "check where a new abstract's methods sit relative to that history."
)

year_summaries = load_year_summaries()
all_year_summaries = load_all_year_summaries()

# ---------------------------------------------------------------------------
# 1. Year-over-year requirement landscape
# ---------------------------------------------------------------------------

st.header("Year-over-Year Requirement Landscape")
st.plotly_chart(
    plot_landscape(year_summaries, all_year_summaries.get("turning_points")),
    use_container_width=True,
)
st.caption("Dotted vertical lines mark turning points — hover for details.")

# ---------------------------------------------------------------------------
# 2. Turning points / persistent / year-specific / unexplained
# ---------------------------------------------------------------------------

st.header("Shifts and Patterns")

with st.expander("Narrative summary (2003–2026)", expanded=True):
    st.write(all_year_summaries.get("narrative", "No narrative available."))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Turning Points")
    for tp in all_year_summaries.get("turning_points", []):
        st.markdown(f"**{tp.get('period')}** — {tp.get('change')}")
        st.caption(tp.get("evidence", ""))

    st.subheader("Persistent Requirements")
    for pr in all_year_summaries.get("persistent_requirements", []):
        st.markdown(f"**{pr.get('name')}**")
        st.caption(pr.get("note", ""))

with col2:
    st.subheader("Year-Specific Requirements")
    for ysr in all_year_summaries.get("year_specific_requirements", []):
        st.markdown(f"**{ysr.get('name')}** ({ysr.get('year')})")
        st.caption(ysr.get("note", ""))

    st.subheader("Unexplained Patterns")
    for pattern in all_year_summaries.get("unexplained_patterns", []):
        st.markdown(f"- {pattern}")

# ---------------------------------------------------------------------------
# 3. Per-year drill-down
# ---------------------------------------------------------------------------

st.header("Per-Year Detail")
years_sorted = sorted(year_summaries.keys(), key=int)
selected_year = st.selectbox("Pick a year", years_sorted, index=len(years_sorted) - 1)

year_payload = year_summaries[selected_year]
st.write(year_payload.get("summary", ""))
st.dataframe(
    pd.DataFrame(year_payload.get("requirements", [])),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# 4. Feasibility check for a new abstract — the only reprocessing step
# ---------------------------------------------------------------------------

st.header("Check Feasibility of a New Abstract")
st.caption(
    "Extracts the methods in your abstract, looks up when each one first "
    "appeared in the literature, and shows where similar methods have "
    "already been used in the existing thesis/grant corpus."
)

with st.form("feasibility_form"):
    abstract_input = st.text_area("Abstract text", height=200)
    input_year = st.number_input(
        "Published / submission year (optional — improves the origin-year lookup)",
        min_value=1900, max_value=2100, value=2026, step=1,
    )
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not abstract_input.strip():
        st.warning("Paste an abstract first.")
    else:
        llm = load_llm()
        method_df = load_method_df()
        abstracts_df = load_abstracts_df()
        matcher = build_matcher(method_df)

        with st.spinner("Extracting methods from the abstract..."):
            result = llm.invoke(prompt.format(abstract=abstract_input))
            parsed = AnalysisResult.model_validate(json.loads(result.content))
            input_methods = [
                {"name": req.name, "search_query": req.search_query}
                for req in parsed.requirements
                if req.category == Category.method_technique
            ]

        if not input_methods:
            st.warning("No methods/techniques were extracted from this abstract.")
        else:
            with st.spinner(
                f"Looking up origin years for {len(input_methods)} method(s) on "
                "Semantic Scholar (a few seconds per method)..."
            ):
                queries = [m["search_query"] for m in input_methods]
                origin_years = get_origin_years(queries, end_year=int(input_year))
                for m in input_methods:
                    m["origin_year"] = origin_years.get(m["search_query"])

            fig = plot_method_adoption_timeline_plotly(
                abstracts_df,
                input_methods,
                matcher,
                year_col="Published Year",
                level_col="Academic Level",
                matched_year_col="year",
                matched_level_col="academic_level",
                matched_title_col="source_title",
                input_year=int(input_year),
                min_similarity=0.45,
            )

            st.session_state["last_result"] = {
                "overall_suitable_degree": parsed.overall_suitable_degree,
                "input_methods": input_methods,
                "fig": fig,
            }

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader(f"Suggested degree level: {result['overall_suitable_degree']}")
    st.dataframe(pd.DataFrame(result["input_methods"]), use_container_width=True, hide_index=True)
    st.plotly_chart(result["fig"], use_container_width=True)
