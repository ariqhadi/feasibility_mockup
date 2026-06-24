"""
build_field_data.py
--------------------
Generate `field-data.js` for the Field Shift Explorer from the REAL data folders.

The web reflects whatever fields live under `data/`. Each field is a folder:

    data/<field>/raw_data.csv               -> per-year, per-level document counts
    data/<field>/per_year_summaries.json    -> per-year story + per-year requirement chips
    data/<field>/all_year_summaries.json    -> turning points (-> eras) + narrative

EVERYTHING the explorer renders is derived from these files — counts, the
PhD/MSc/grant lines, the per-year stories and requirement chips, the turning
points, and the visual scaffolding (eras + method/data/compute "lifelines",
maturity, tagline). There is no hand-authored field data: add or edit a folder
under data/, re-run this script, refresh the page, and the field appears.

Run:  python3 build_field_data.py   ->  writes field-data.js
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

SPAN = (2003, 2026)           # the explorer's chart axis is fixed to this range
YEARS = list(range(SPAN[0], SPAN[1] + 1))

DATA_DIR = "data"
OUT_PATH = "field-data.js"

# Put sentiment_analysis first if present, otherwise alphabetical.
PREFERRED_FIRST = "sentiment_analysis"

# ---- colour palettes (cool -> warm; the newest era is always the warm accent) ----
ERA_PALETTE = ["#c9a14a", "#5b8a72", "#4a78c2", "#8a5fb0", "#3f8a9a", "#b08a3e"]
ACCENT = "#cf5b3e"
METHOD_COLORS = ["#4f7a6b", "#3f6ea5", "#7a5aa0", "#b8553a", "#b08a3e", "#9aa0a6"]
DATA_COLORS = ["#5a8a7a", "#3f9a84", "#3f8a9a", "#5f9aa8", "#7aa0b0"]
COMPUTE_COLORS = ["#c7a14a", "#b8862f", "#cf8a3e"]

# requirement category -> chip group shown in the per-year detail panel
CAT_TO_GROUP = {
    "data": "Data",
    "method_technique": "Methods",
    "tool_library": "Tools",
    "compute": "Compute",
    "human_effort": "Human",
}

_STOP = {"and", "or", "the", "of", "for", "a", "an", "to", "in", "on",
         "with", "using", "based", "via", "as"}


# --------------------------------------------------------------------------- #
# low-level readers
# --------------------------------------------------------------------------- #
def read_counts(csv_path):
    """Distinct documents per year, split into PhD / MSc(=MA) / grant."""
    phd = defaultdict(int)
    msc = defaultdict(int)
    grant = defaultdict(int)
    seen = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            did = (row.get("data_id") or "").strip()
            if not did or did in seen:
                continue
            seen.add(did)
            yr = (row.get("Published Year") or "").strip()
            if not yr.isdigit():
                continue
            y = int(yr)
            if not (SPAN[0] <= y <= SPAN[1]):
                continue
            level = (row.get("Academic Level") or "").strip().lower()
            if level == "grant":
                grant[y] += 1
            elif level == "phd":
                phd[y] += 1
            else:  # MA / MSc / blank thesis -> MSc level
                msc[y] += 1
    arr = lambda d: [d.get(y, 0) for y in YEARS]
    return arr(phd), arr(msc), arr(grant)


def read_per_year(path):
    """(stories{year->text}, year_reqs{year->[{name,group}]}, reqs_by_cat)."""
    with open(path, encoding="utf-8") as f:
        per_year = json.load(f)

    stories, year_reqs = {}, {}
    # reqs_by_cat[category][group_key] = {"label", "first", "years": set()}
    reqs_by_cat = defaultdict(lambda: defaultdict(lambda: {"label": None, "first": 9999, "years": set()}))

    for ys, payload in per_year.items():
        if not ys.isdigit():
            continue
        y = int(ys)
        if not (SPAN[0] <= y <= SPAN[1]):
            continue
        summ = (payload.get("summary") or "").strip()
        if summ:
            stories[y] = summ
        chips = []
        for req in payload.get("requirements", []):
            cat = req.get("category", "")
            name = (req.get("name") or "").strip()
            if not name:
                continue
            group = CAT_TO_GROUP.get(cat)
            if group:
                chips.append({"name": name, "group": group})
            # accumulate canonical families for lifelines
            key = _family_key(name)
            fam = reqs_by_cat[cat][key]
            cand = _short_label(name)
            if fam["label"] is None or len(cand) < len(fam["label"]):
                fam["label"] = cand
            fam["first"] = min(fam["first"], y)
            fam["years"].add(y)
        if chips:
            year_reqs[y] = chips
    return stories, year_reqs, reqs_by_cat


def read_turning_points(path):
    with open(path, encoding="utf-8") as f:
        allyr = json.load(f)
    tps = []
    for tp in allyr.get("turning_points", []):
        p = str(tp.get("period", "")).strip()
        if not p.isdigit():
            continue
        y = int(p)
        if SPAN[0] <= y <= SPAN[1]:
            tps.append({"y": y, "change": (tp.get("change") or "").strip()})
    tps.sort(key=lambda d: d["y"])
    return tps, (allyr.get("narrative") or "").strip()


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def _family_key(name):
    """Collapse near-duplicate requirement names to a family: first two
    significant words of the (parenthetical-stripped) lowercased name."""
    n = re.sub(r"\([^)]*\)", " ", name.lower())
    n = re.sub(r"[^a-z0-9\s/]", " ", n)
    words = [w for w in n.split() if w and w not in _STOP]
    return " ".join(words[:2]) if words else name.lower().strip()


def _short_label(name, limit=30):
    """Trim a requirement name to a compact chart label."""
    s = re.sub(r"\s*\([^)]*\)", "", name).strip()  # drop parentheticals
    s = re.sub(r"\s+", " ", s)
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def _tp_label(change):
    """Short tick/era label from a turning-point change sentence."""
    # prefer an acronym in parentheses, e.g. "... (LLMs) ..."
    m = re.search(r"\(([A-Za-z]{2,5})s?\)", change)
    if m:
        return m.group(1)
    # else the subject phrase before the first change verb
    head = re.split(r"\b(first|becomes?|become|emerges?|emerge|appears?|appear|arrives?|arrive)\b",
                    change, maxsplit=1)[0].strip()
    head = re.sub(r"\s*\([^)]*\)", "", head)
    words = head.split()[:3]
    # don't end a label on a dangling connector ("GPU compute for" -> "GPU compute")
    while len(words) > 1 and words[-1].lower() in _STOP:
        words.pop()
    return " ".join(words) if words else change[:14]


# --------------------------------------------------------------------------- #
# derivations
# --------------------------------------------------------------------------- #
def build_eras(tps, narrative):
    """Eras = the spans between turning points. Each era is named after the
    shift that opens it; the first ('foundational') era covers the run-up."""
    boundaries = [SPAN[0]] + [t["y"] for t in tps] + [SPAN[1] + 0.2]
    eras = []
    n = len(boundaries) - 1
    for i in range(n):
        a, b = boundaries[i], boundaries[i + 1]
        if i == 0:
            name = "Foundational"
            note = "The field's run-up — before the first major shift in its method stack."
        else:
            tp = tps[i - 1]
            name = _tp_label(tp["change"])
            note = tp["change"]
        color = ACCENT if i == n - 1 else ERA_PALETTE[i % len(ERA_PALETTE)]
        eras.append({"name": name, "a": a, "b": b, "c": color, "note": note})
    return eras


def _pick_families(reqs_by_cat, cat, colors, top_n):
    """Top requirement families in a category -> lifeline entries, ranked by
    recurrence (distinct years seen) then earliest appearance."""
    fams = reqs_by_cat.get(cat, {})
    ranked = sorted(
        fams.values(),
        key=lambda d: (-len(d["years"]), d["first"], d["label"] or ""),
    )[:top_n]
    # stable, readable order on the chart: by first-appearance year
    ranked.sort(key=lambda d: d["first"])
    out = []
    for i, fam in enumerate(ranked):
        last = max(fam["years"])
        end = SPAN[1] if last >= SPAN[1] - 2 else last
        out.append({
            "label": fam["label"],
            "s": fam["first"],
            "e": end,
            "color": colors[i % len(colors)],
            "op": 0.85,
        })
    return out


def build_lifelines(reqs_by_cat):
    return {
        "methods": _pick_families(reqs_by_cat, "method_technique", METHOD_COLORS, 6),
        "data": _pick_families(reqs_by_cat, "data", DATA_COLORS, 5),
        "compute": _pick_families(reqs_by_cat, "compute", COMPUTE_COLORS, 3),
    }


def derive_maturity(eras, thesis, grant):
    """Heuristic 0..1: more method generations + sustained recent volume read
    as a more developed field. Only affects the Mature/Maturing/Emerging word."""
    gens = min(1.0, (len(eras) - 1) / 5.0)  # 0 (no shifts) .. 1 (5+ shifts)
    total = sum(thesis) + sum(grant)
    recent = sum(thesis[-6:]) + sum(grant[-6:])
    sustain = (recent / total) if total else 0.0  # share of activity in last 6 yrs
    # a field still ramping hard (high recent share) is less "settled"
    settle = max(0.0, 1.0 - abs(sustain - 0.25) * 1.5)
    return round(min(0.95, 0.45 * gens + 0.55 * settle), 2)


def build_field(folder):
    base = os.path.join(DATA_DIR, folder)
    phd, msc, grant = read_counts(os.path.join(base, "raw_data.csv"))
    thesis = [phd[i] + msc[i] for i in range(len(YEARS))]
    stories, year_reqs, reqs_by_cat = read_per_year(os.path.join(base, "per_year_summaries.json"))
    tps, narrative = read_turning_points(os.path.join(base, "all_year_summaries.json"))

    eras = build_eras(tps, narrative)
    lifelines = build_lifelines(reqs_by_cat)
    maturity = derive_maturity(eras, thesis, grant)
    label = folder.replace("_", " ").title()
    last_era = eras[-1]["name"] if eras else "the current stack"
    tagline = f"{len(eras)} method generations across {SPAN[0]}–{SPAN[1]} — now centred on {last_era}."

    return {
        "label": label,
        "maturity": maturity,
        "tagline": tagline,
        "eras": eras,
        "turningPoints": [{"y": t["y"], "t": _tp_label(t["change"])} for t in tps],
        "volume": {"thesis": thesis, "grant": grant},
        "phd": phd,
        "msc": msc,
        "lifelines": lifelines,
        "constants": [],
        "years": {str(y): stories[y] for y in sorted(stories)},
        "yearReqs": {str(y): year_reqs[y] for y in sorted(year_reqs)},
    }


def discover_fields():
    if not os.path.isdir(DATA_DIR):
        return []
    fields = []
    for name in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, name)
        if not os.path.isdir(d):
            continue
        needed = ["raw_data.csv", "per_year_summaries.json", "all_year_summaries.json"]
        if all(os.path.exists(os.path.join(d, f)) for f in needed):
            fields.append(name)
        else:
            print(f"  ! skipping data/{name} — missing one of {needed}")
    # preferred field first
    if PREFERRED_FIRST in fields:
        fields.remove(PREFERRED_FIRST)
        fields.insert(0, PREFERRED_FIRST)
    return fields


def main():
    folders = discover_fields()
    if not folders:
        raise SystemExit(f"No valid field folders found under ./{DATA_DIR}/")

    fields = {key: build_field(key) for key in folders}

    blocks = []
    for key in folders:
        js = json.dumps(fields[key], ensure_ascii=False, indent=4)
        js = "\n".join(("    " + ln) if ln else ln for ln in js.splitlines())
        blocks.append(f"  {key}:\n{js}")

    order = "[" + ",".join(f"'{k}'" for k in folders) + "]"
    header = (
        "// Input data for the Field Shift Explorer.\n"
        "// GENERATED by build_field_data.py — do not edit by hand.\n"
        "// Every field below is derived entirely from its folder under ./data/:\n"
        "//   data/<field>/raw_data.csv               (per-year, per-level document counts)\n"
        "//   data/<field>/per_year_summaries.json    (per-year story + requirement chips)\n"
        "//   data/<field>/all_year_summaries.json    (turning points -> eras, narrative)\n"
        "// To change what the web shows: add/edit a folder under data/, then run\n"
        "//   python3 build_field_data.py\n"
        "// Counts (phd/msc/grant) and stories are real; eras, lifelines, maturity and\n"
        "// tagline are derived deterministically from the same files.\n\n"
    )
    body = (
        header
        + f"export const SPAN = [{SPAN[0]}, {SPAN[1]}];\n\n"
        + "export const FIELDS = {\n\n"
        + ",\n\n".join(blocks)
        + "\n\n};\n\n"
        + f"export const FIELD_ORDER = {order};\n"
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(body)

    print(f"Wrote {OUT_PATH} with {len(folders)} field(s): {', '.join(folders)}")
    for key in folders:
        fd = fields[key]
        print(f"  [{key}] theses={sum(fd['volume']['thesis'])} "
              f"(PhD {sum(fd['phd'])} / MSc {sum(fd['msc'])}), "
              f"grants={sum(fd['volume']['grant'])}, "
              f"eras={len(fd['eras'])}, "
              f"methods={len(fd['lifelines']['methods'])}, "
              f"years={len(fd['years'])}, maturity={fd['maturity']}")


if __name__ == "__main__":
    main()
