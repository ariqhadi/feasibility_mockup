"""
feasibility.py
--------------
"Test an abstract for feasibility" at a given year, using the dependency-window
model. No offline fallbacks: if extraction or dating cannot be performed, the
function RAISES (FeasibilityError) rather than returning a degraded guess.

Pipeline:
  1. EXTRACT dependencies from the abstract via the LLM (structured requirements).
  2. FILTER to true, datable prerequisites:
        - drop produced_by_study (outputs the work creates)
        - keep binding_candidate (items that can decide the year)
  3. RESOLVE each kept dependency to an AVAILABILITY WINDOW [available_from,
     available_until]:
        - access / data  -> date_resolver (live web search; can detect CLOSURE)
        - everything else -> Semantic Scholar (method_history); one-sided window
  4. VERDICT by window CONTAINMENT:
        target_year must be >= available_from AND (until is None or <= until).

Public result objects (MethodResult / FeasibilityResult) and the CLI/JSON output
are UNCHANGED so the existing UI / Flask layer keeps working. `origin` carries
available_from; available_until / closure_type / dependency_type / reason are
additive optional fields.

CLI:
    python feasibility.py --abstract path/to/abstract.txt --year 2018
    echo "we fine-tune BERT on tweets" | python feasibility.py --year 2015
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Reuse reusable_parts/ (method_history, tools) and the sibling date_resolver.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REUSABLE = os.path.join(_HERE, "reusable_parts")
if _REUSABLE not in sys.path:
    sys.path.insert(0, _REUSABLE)


class FeasibilityError(RuntimeError):
    """Raised when a required step (extraction or dating) cannot be completed.
    The Flask layer should catch this and surface it to the user."""


# --------------------------------------------------------------------------- #
# 1. Dependency extraction (LLM only)
# --------------------------------------------------------------------------- #
EXTRACT_PROMPT = r"""You are an experienced research supervisor with 20+ years of experience across Computer Science and adjacent empirical fields.

Your task is to read a thesis/paper abstract and extract the **dependencies** the described research relies on - the concrete things that must exist or be obtainable for the work to be carried out. The goal is downstream feasibility-by-year analysis, so you must surface *what each requirement depends on*, not just name the technique.

Core principle: **techniques are timeless; dependencies are dated.** What dates a study is the data it needs, the real-world event it assumes already happened, or the tool/material/access required to obtain that data. Extract those.

### Hard rules (anti-hallucination)
1. Ground everything in a verbatim `evidence_span` from the abstract; if you cannot, omit it.
2. Do NOT supply dates. Your job is *what*, not *when*.
3. Prefer omission over invention.
4. Use "unknown" rather than guessing.
5. Mark `inferred: true` only when implied, not named outright.
6. No author names in search queries.
7. Motivational framing ("With the advancement of X...") is NOT a dependency.
8. Things the study CREATES are outputs: mark `produced_by_study: true`, never `binding_candidate: true`.

### Per-requirement fields
category, name, description, evidence_span,
dependency_type (data|event|instrument|material|method|compute|access),
hard_or_soft, binding_candidate, produced_by_study, inferred,
estimated_difficulty, search_query.

Return ONLY valid JSON of the form:
{{"overall_suitable_degree": "...", "requirements": [ {{ ...fields... }} ]}}

[THESIS ABSTRACT]
{abstract}
[/THESIS ABSTRACT]
"""

_MAX_DEPS = 10

# dependency_types whose availability window can CLOSE -> live closure check.
_CLOSEABLE_TYPES = {"access", "data"}

# dependency_types dated via the web resolver (date_resolver): closeable ones
# plus events (a real-world occurrence isn't in the academic literature, so it
# must be dated by web search, not Semantic Scholar).
_WEB_TYPES = _CLOSEABLE_TYPES | {"event"}

# data/access resources dated at or before this are almost certainly a model
# floor/sentinel (e.g. 1945), not a real availability date -> treated as undated.
_IMPLAUSIBLE_FLOOR = 1995


@dataclass
class _Dep:
    """Internal richer dependency record (not exposed to the UI)."""
    name: str
    dependency_type: str = "method"
    produced_by_study: bool = False
    binding_candidate: bool = True
    search_query: Optional[str] = None
    evidence_span: str = ""

    def as_resolver_dict(self) -> dict:
        """Shape expected by date_resolver.resolve_date()."""
        return {
            "name": self.name,
            "dependency_type": self.dependency_type,
            "produced_by_study": self.produced_by_study,
            "binding_candidate": self.binding_candidate,
            "search_query": self.search_query or self.name,
            "evidence_span": self.evidence_span,
        }


def _parse_requirements(raw: str) -> List[_Dep]:
    """Parse the LLM's {"requirements":[...]} payload into _Dep records.
    Tolerant of a bare array, an object with a requirements key, and code fences.
    Raises FeasibilityError if nothing usable is found."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", cleaned)
    if not m:
        raise FeasibilityError("LLM response contained no JSON.")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise FeasibilityError(f"LLM returned invalid JSON: {exc}") from exc

    reqs = data.get("requirements", data) if isinstance(data, dict) else data
    if not isinstance(reqs, list) or not reqs:
        raise FeasibilityError("LLM response had no requirements array.")

    deps: List[_Dep] = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").strip()
        if not name:
            continue
        deps.append(_Dep(
            name=name,
            dependency_type=(r.get("dependency_type") or "method").strip().lower(),
            produced_by_study=bool(r.get("produced_by_study", False)),
            binding_candidate=bool(r.get("binding_candidate", True)),
            search_query=(r.get("search_query") or name).strip(),
            evidence_span=(r.get("evidence_span") or "").strip(),
        ))
    if not deps:
        raise FeasibilityError("LLM returned no usable requirements.")
    return deps


def extract_dependencies(text: str) -> List[_Dep]:
    """LLM dependency extraction via reusable_parts/tools.get_model().
    Raises FeasibilityError on any failure (no key, model error, bad JSON)."""
    try:
        from tools import get_model
    except Exception as exc:  # noqa: BLE001
        raise FeasibilityError(f"Could not import the model layer: {exc}") from exc

    llm = get_model()
    if llm is None:
        raise FeasibilityError("get_model() returned None (check reusable_parts/config.json).")
    try:
        resp = llm.invoke(EXTRACT_PROMPT.format(abstract=text))
    except Exception as exc:  # noqa: BLE001
        raise FeasibilityError(f"LLM extraction call failed: {exc}") from exc
    content = getattr(resp, "content", resp)
    return _parse_requirements(content)[:_MAX_DEPS]


def _keep_for_dating(deps: List[_Dep]) -> List[_Dep]:
    """Filter to true, datable prerequisites: drop outputs, keep binding ones.
    Raises FeasibilityError if nothing datable remains."""
    kept = [d for d in deps if not d.produced_by_study and d.binding_candidate]
    if not kept:
        # fall back to non-output deps before giving up entirely
        kept = [d for d in deps if not d.produced_by_study]
    if not kept:
        raise FeasibilityError(
            "No datable dependencies found (all extracted items were study outputs)."
        )
    return kept


# --------------------------------------------------------------------------- #
# 2. Availability-window lookup (live: Semantic Scholar + date_resolver)
# --------------------------------------------------------------------------- #
try:
    from method_history import get_origin_year as _ss_origin_year  # type: ignore
except Exception:  # pragma: no cover
    _ss_origin_year = None

try:
    import date_resolver as _dr  # sibling module; uses ChatOpenRouter web plugin
    _dr_import_error = None
except Exception as exc:  # pragma: no cover
    _dr = None
    _dr_import_error = exc


def _resolve_window(dep: _Dep, end_year: Optional[int]) -> dict:
    """Return a window dict: {from, until, closure_type, source}.
    access/data/event -> date_resolver (live web). access/data are closure-aware;
        event is one-sided (handled inside date_resolver by dependency_type).
    method/material/instrument/compute -> Semantic Scholar (one-sided).
    Raises FeasibilityError if the required resolver is missing or yields nothing."""
    if dep.dependency_type in _WEB_TYPES:
        if _dr is None:
            raise FeasibilityError(
                f"date_resolver unavailable ({_dr_import_error}); "
                f"cannot date {dep.dependency_type} dependency '{dep.name}'."
            )
        finding = _dr.resolve_date(dep.as_resolver_dict())
        if finding.available_from is None:
            raise FeasibilityError(
                f"Could not determine availability window for '{dep.name}'."
            )
        window = {
            "from": finding.available_from,
            "until": finding.available_until,
            "closure_type": None if finding.closure_type == "none" else finding.closure_type,
            "source": finding.source_url or "web",
        }
        # floor sanity guard applies only to data/access (where a 1945-style
        # sentinel is implausible); an event can legitimately be any year.
        if dep.dependency_type in _CLOSEABLE_TYPES:
            window = _sanitize_window(window)
        return window

    # one-sided types: Semantic Scholar
    if _ss_origin_year is None:
        raise FeasibilityError(
            f"Semantic Scholar layer unavailable; cannot date '{dep.name}'."
        )
    try:
        yr = _ss_origin_year(dep.search_query or dep.name, end_year=end_year, retries=1)
    except Exception as exc:  # noqa: BLE001
        raise FeasibilityError(f"Origin lookup failed for '{dep.name}': {exc}") from exc
    if not yr or not (1945 <= yr <= 2026):
        raise FeasibilityError(f"No plausible origin year found for '{dep.name}'.")
    return {"from": yr, "until": None, "closure_type": None, "source": "Semantic Scholar"}


# --------------------------------------------------------------------------- #
# 3. Orchestration + verdict  (output shape preserved)
# --------------------------------------------------------------------------- #
@dataclass
class MethodResult:
    name: str
    origin: int                       # == available_from (UNCHANGED field name)
    source: str
    infeasible: bool
    # additive, optional -> existing UI ignores these safely
    available_until: Optional[int] = None
    closure_type: Optional[str] = None
    dependency_type: str = "method"
    reason: Optional[str] = None


@dataclass
class FeasibilityResult:
    year: int
    methods: List[MethodResult]
    extraction: str
    infeasible: bool
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def _date_dep(dep: _Dep, year: int) -> MethodResult:
    """Resolve a dependency's window only (no year check yet). The containment
    verdict is applied later, after same-resource facets are merged."""
    w = _resolve_window(dep, end_year=year)
    return MethodResult(
        name=dep.name, origin=w["from"], source=w["source"], infeasible=False,
        available_until=w["until"], closure_type=w["closure_type"],
        dependency_type=dep.dependency_type, reason=None,
    )


def _apply_year(m: MethodResult, year: int) -> MethodResult:
    """Run the window-containment check and set infeasible/reason in place."""
    frm, until, ctype = m.origin, m.available_until, m.closure_type
    if year < frm:
        m.infeasible, m.reason = True, f"not available until {frm}"
    elif until is not None and year > until:
        if ctype in ("discontinued", "restricted"):
            m.infeasible, m.reason = True, f"access {ctype} in {until}"
        elif ctype == "price_raised":
            m.infeasible, m.reason = True, f"access restricted/costly after {until}"
    return m


# Tokens that, when shared between two dependency names/queries, mark them as
# facets of the SAME underlying resource (so they should be dated once, not
# evaluated independently). Extend as needed for other recurring platforms.
_SAME_RESOURCE_HINTS = [
    {"twitter", "tweet", "tweets", "x"},      # Twitter/X data + API access
]


def _resource_key(dep: _Dep) -> Optional[frozenset]:
    """Return the hint-set a dependency belongs to, or None if it matches none.
    Used only for access/data deps, where redundant facets cause split windows."""
    if dep.dependency_type not in _CLOSEABLE_TYPES:
        return None
    words = set(re.findall(r"[a-z0-9]+", f"{dep.name} {dep.search_query}".lower()))
    for hint in _SAME_RESOURCE_HINTS:
        if words & hint:
            return frozenset(hint)
    return None


def _merge_group(members: List[MethodResult]) -> MethodResult:
    """Combine several MethodResults that are facets of one resource into a
    single window by CONTAINMENT: the tightest window all facets agree on.
      available_from -> the LATEST start (must satisfy the most restrictive facet)
      available_until -> the EARLIEST close (window shuts as soon as any facet does)
    This biases toward the precise binding mechanism (e.g. the API tier's 2021)
    rather than 'the platform has existed since 2006'."""
    frm = max(m.origin for m in members)
    closes = [m.available_until for m in members if m.available_until is not None]
    until = min(closes) if closes else None
    # carry the closure_type of whichever facet defines the earliest close
    ctype = None
    if until is not None:
        ctype = next((m.closure_type for m in members if m.available_until == until), None)
    name = " / ".join(sorted({m.name for m in members}))[:120]
    src = next((m.source for m in members if m.origin == frm), members[0].source)
    return MethodResult(
        name=name, origin=frm, source=src, infeasible=False,
        available_until=until, closure_type=ctype,
        dependency_type=members[0].dependency_type, reason=None,
    )


def _sanitize_window(w: dict) -> dict:
    """Guard against sentinel/floor years leaking through as if they were real
    findings. A start year at or below the implausible floor for a data/access
    resource is treated as unknown rather than a dated result."""
    frm = w.get("from")
    if frm is not None and frm <= _IMPLAUSIBLE_FLOOR:
        # don't trust 'available since 1945' for a modern data/access dependency
        raise FeasibilityError(
            f"resolver returned an implausible origin year ({frm}); treating as undated"
        )
    return w


def analyze(abstract: str, year: int) -> FeasibilityResult:
    """Run the full feasibility test for one abstract at a given year.
    Raises FeasibilityError if extraction or dating cannot be completed.

    Order matters: we DATE every dependency first, then MERGE facets of the same
    resource by window-containment, then apply the year check. Merging after
    dating (not before) lets us reconcile windows correctly instead of guessing
    which facet to keep."""
    text = (abstract or "").strip()
    if len(text) < 20:
        raise ValueError("Paste an abstract (at least a sentence or two) first.")

    deps = _keep_for_dating(extract_dependencies(text))

    from concurrent.futures import ThreadPoolExecutor

    # 1. DATE every dependency (window only). I/O-bound -> parallel.
    dated: List[tuple[_Dep, MethodResult]] = []
    skipped: List[str] = []
    with ThreadPoolExecutor(max_workers=min(8, len(deps))) as ex:
        futures = {ex.submit(_date_dep, d, year): d for d in deps}
        for fut, d in futures.items():
            try:
                dated.append((d, fut.result()))
            except FeasibilityError as exc:
                skipped.append(f"{d.name} ({exc})")

    if not dated:
        raise FeasibilityError("Could not date any dependency. " + " ".join(skipped))

    # 2. MERGE same-resource facets by tightest window.
    groups: dict = {}
    singles: List[MethodResult] = []
    for dep, mr in dated:
        key = _resource_key(dep)
        if key is None:
            singles.append(mr)
        else:
            groups.setdefault(key, []).append(mr)

    merged_away: List[str] = []
    methods: List[MethodResult] = list(singles)
    for key, members in groups.items():
        if len(members) == 1:
            methods.append(members[0])
        else:
            merged = _merge_group(members)
            merged_away.extend(m.name for m in members)
            methods.append(merged)

    # 3. APPLY the year check to the final (merged) windows.
    for m in methods:
        _apply_year(m, year)

    methods.sort(key=lambda m: m.origin)
    late = [m for m in methods if m.infeasible]
    infeasible = bool(late)
    if infeasible:
        detail = ", ".join(
            f"{m.name} ({m.reason})" if m.reason else f"{m.name} ({m.origin})"
            for m in late
        )
        verdict = (
            f"Infeasible for {year}: {len(late)} dependenc"
            f"{'ies' if len(late) > 1 else 'y'} out of window - {detail}."
        )
    else:
        verdict = f"Plausible for {year}: all {len(methods)} binding dependencies available then."
    if merged_away:
        verdict += f" (Merged {len(merged_away)} facet(s) of shared resources.)"
    if skipped:
        verdict += f" (Could not date {len(skipped)}: {'; '.join(skipped)}.)"

    return FeasibilityResult(
        year=year, methods=methods, extraction="llm",
        infeasible=infeasible, verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Test an abstract for feasibility at a given year.")
    p.add_argument("--abstract", help="Path to a file with the abstract text. Omit to read stdin.")
    p.add_argument("--year", type=int, required=True, help="Year to test the abstract against.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = p.parse_args(argv)

    if args.abstract:
        with open(args.abstract, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    try:
        res = analyze(text, args.year)
    except (ValueError, FeasibilityError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    print(f"Detected dependencies ({len(res.methods)})  [extraction: {res.extraction}]:")
    for m in res.methods:
        flag = "X INFEASIBLE" if m.infeasible else "OK"
        window = f"{m.origin}" + (f"-{m.available_until}" if m.available_until else "+")
        print(f"  {m.name:34s} {window:>10s}  [{m.source}]  {flag}")
    print(f"\nVerdict: {res.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())