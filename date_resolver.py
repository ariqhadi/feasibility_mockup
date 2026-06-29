"""
Date resolver — returns an AVAILABILITY WINDOW per dependency, not a single year.

Why a window: access can close (Twitter API affordable tier ~2013, shut for
researchers ~2023). Feasibility is window-CONTAINMENT, not "after the start date".

Routing by dependency_type:
  web + closure-aware  : access, data        (can become unavailable)
  web + one-sided      : event               (happens once, never "un-happens")
  semantic scholar     : method, material, instrument, compute
                         (once they exist they stay available -> until = null)
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
import os

SEED = 42
MODEL = "anthropic/claude-sonnet-4-5"

ClosureType = Literal["discontinued", "restricted", "price_raised", "none"]


class DateFinding(BaseModel):
    available_from: Optional[int] = Field(
        description="Earliest year it existed / became available. null if not found."
    )
    available_until: Optional[int] = Field(
        description="Year it stopped being available. null if still available."
    )
    closure_type: ClosureType = Field(
        description="How it ended: discontinued / restricted / price_raised / none."
    )
    confidence: Literal["high", "medium", "low", "unknown"]
    source_url: Optional[str]


# ---- model instance: web plugin, structured output -------------------------
_web_model = ChatOpenRouter(
    model=MODEL, temperature=0, seed=SEED, max_tokens=400,
    plugins=[{"id": "web", "max_results": 3}],
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
).with_structured_output(DateFinding)

# Semantic Scholar resolver lives in a separate module; wrap it to DateFinding.
# from semantic_scholar_resolver import ss_lookup   # -> returns (year, url)


# ---- prompts: closure-aware vs one-sided -----------------------------------
_CLOSURE_PROMPT = """Determine the AVAILABILITY WINDOW of the thing below.
Return ONLY the structured fields. Do not explain.

- available_from: earliest year it became available.
- available_until: year it STOPPED being available - search specifically for whether
  access was later revoked, the API tier discontinued, the corpus taken down, or the
  price raised beyond ordinary research reach. If still available, set null.
- closure_type: discontinued / restricted / price_raised / none.
- If web results don't clearly support a year, use null and confidence="unknown".
  Do NOT guess from prior knowledge. source_url must come from the results or be null.

Thing ({dep_type}): {query}
Context (disambiguation only): {evidence_span}
"""

_ONE_SIDED_PROMPT = """Determine the year the thing below FIRST occurred or existed.
Return ONLY the structured fields. Do not explain.

- available_from: the year it happened / first existed.
- available_until: null (this kind of thing does not become unavailable).
- closure_type: "none".
- If web results don't clearly support a year, use null and confidence="unknown".
  Do NOT guess from prior knowledge. source_url must come from the results or be null.

Thing ({dep_type}): {query}
Context (disambiguation only): {evidence_span}
"""

_CLOSURE_TYPES = {"access", "data"}     # web + ask about closure
_ONE_SIDED_WEB = {"event"}              # web, one-sided
_SCHOLAR_TYPES = {"method", "material", "instrument", "compute"}


def _skip(reason="unknown") -> DateFinding:
    return DateFinding(available_from=None, available_until=None,
                       closure_type="none", confidence=reason, source_url=None)


def resolve_date(dep: dict) -> DateFinding:
    # pre-filters: nothing to date
    if dep.get("produced_by_study"):
        return _skip()
    if not dep.get("binding_candidate", False):
        return _skip()

    dtype = dep["dependency_type"]
    query = dep.get("search_query") or dep["name"]
    span = dep.get("evidence_span", "")

    if dtype in _SCHOLAR_TYPES:
        # year, url = ss_lookup(query)
        # return DateFinding(available_from=year, available_until=None,
        #                    closure_type="none",
        #                    confidence="high" if year else "unknown",
        #                    source_url=url)
        raise NotImplementedError("route to Semantic Scholar resolver")

    if dtype in _CLOSURE_TYPES:
        prompt = _CLOSURE_PROMPT
    elif dtype in _ONE_SIDED_WEB:
        prompt = _ONE_SIDED_PROMPT
    else:
        raise ValueError(f"unhandled dependency_type: {dtype}")

    return _web_model.invoke(prompt.format(dep_type=dtype, query=query, evidence_span=span))


# ---- feasibility = window containment, in pure Python ----------------------
def is_feasible(findings: list[DateFinding], target_year: int) -> str:
    strained = False
    for f in findings:
        if f.available_from is None:
            continue                                   # undatable; skip/flag separately
        if target_year < f.available_from:
            return "infeasible"                        # not invented yet
        if f.available_until is not None and target_year > f.available_until:
            if f.closure_type in ("discontinued", "restricted"):
                return "infeasible"                    # window closed hard
            if f.closure_type == "price_raised":
                strained = True                        # possible but costly
    return "strained" if strained else "feasible"