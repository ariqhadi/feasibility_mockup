"""
feasibility.py
--------------
"Test an abstract for feasibility" — the Python port of the logic that used to
live inline in `Field Shift Explorer v2.dc.html`.

Given an abstract and a year, it:
  1. EXTRACTS the methods/techniques the abstract uses.
  2. Looks up each method's ORIGIN YEAR (when it first appears in the literature).
  3. Returns an anachronism VERDICT: any method whose origin postdates the
     abstract's year makes the abstract anachronistic for that year.

This reuses the existing `reusable_parts` building blocks rather than
reimplementing them:
  - method extraction  -> the LLM prompt + Pydantic schema from program.ipynb,
                          driven by reusable_parts/tools.get_model()
  - origin-year lookup -> reusable_parts/method_history.get_origin_year()
                          (robust Semantic Scholar bulk search)

Both expensive steps degrade gracefully so the module always returns something:
  - no LLM key / LLM error -> keyword extraction (offline table)
  - no API result          -> local origin-year estimate (offline table)

Edit the two offline tables (KEYWORD_METHODS, LOCAL_ORIGINS) and the EXTRACT_PROMPT
to tune behaviour — they are plain Python data, no build step.

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
# Reuse reusable_parts/ (method_history, tools, prompt). They live in a sibling
# folder and load their own config/.env relative to cwd, so we add them to the
# path and let them manage their own configuration.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REUSABLE = os.path.join(_HERE, "reusable_parts")
if _REUSABLE not in sys.path:
    sys.path.insert(0, _REUSABLE)

try:
    from method_history import get_origin_year as _ss_origin_year  # type: ignore
except Exception:  # pragma: no cover - reusable_parts missing
    _ss_origin_year = None


# --------------------------------------------------------------------------- #
# 1. Method extraction
# --------------------------------------------------------------------------- #

# The extraction prompt, lifted from program.ipynb (cell 1) and trimmed to the
# task we need here: pull out the methods/models/algorithms/data the abstract
# actually uses. {abstract} is filled in at call time.
EXTRACT_PROMPT = """You are an experienced research supervisor with 20+ years of experience across Computer Science and adjacent empirical fields.

Your task is to read a thesis/paper abstract and extract the **dependencies** the described research relies on — the concrete things that must exist or be obtainable for the work to be carried out. The goal is downstream feasibility-by-year analysis, so you must surface *what each requirement depends on*, not just name the technique.

Core principle: **techniques are timeless; dependencies are dated.** A method like "before-after comparison" tells us nothing about when a study became possible. What dates a study is the data it needs, the real-world event it assumes already happened, or the tool/material/access required to obtain that data. Extract those.

---

### Hard rules (anti-hallucination)

1. **Ground everything.** Every requirement MUST include an `evidence_span`: a verbatim substring copied exactly from the abstract that supports it. If you cannot point to a span, do not include the requirement.
2. **Do NOT supply dates.** Never state when something was invented, released, or became available. Your job is *what*, not *when*. Dates are resolved later by retrieval.
3. **Prefer omission over invention.** If a requirement is uncertain or unsupported, leave it out. Do not pad the list with plausible-sounding resources.
4. **Use `unknown` freely.** For any field you are unsure of, output "unknown" rather than guessing.
5. **Mark inference.** If a dependency is implied but not explicitly stated, include it only with `"inferred": true` and still attach the closest supporting span. Explicit dependencies are `"inferred": false`. If something is named outright in the text, it is NOT inferred.
6. **No author names in search queries.** Focus queries on the method, tool, dataset, event, or material itself.
7. **Motivational framing is NOT a dependency.** Background/scene-setting sentences ("With the advancement of X…", "As Y has grown in popularity…", "In the era of Z…") explain why the research matters; they are not things the study depends on. Do NOT extract them as dependencies — especially not as `event` / `hard` / `binding_candidate: true`, which would wrongly peg the feasibility year to that backdrop. Only treat a phenomenon as a dependency if the study's data, samples, or comparison are literally drawn from or defined relative to it. (Contrast: "ownership change in October 2022" IS a dependency because the before/after comparison is defined relative to it; "with the advancement of the World Wide Web" is NOT, because the study merely sits in that context.)
8. **Distinguish prerequisites from outputs.** Things the study *creates* ("a lexicon created in this work", "we construct a dataset", "our proposed model") are OUTPUTS, not prerequisites — the study does not depend on them existing beforehand. Mark these `"produced_by_study": true`. They must NOT be `binding_candidate: true` and should be excluded from feasibility dating. Only resources that must pre-exist for the work to begin are true dependencies.

---

### Per-requirement fields

For each dependency, output:

- `category`: "data" | "method_technique" | "tool_library" | "compute" | "human_effort" | "other"
- `name`: short clear name of the resource.
- `description`: one sentence on how it is used in the work.
- `evidence_span`: verbatim substring from the abstract supporting this requirement.
- `dependency_type`: "data" | "event" | "instrument" | "material" | "method" | "compute" | "access"
    - data = a dataset/corpus/records that must exist
    - event = a real-world occurrence the study assumes already happened (e.g. an ownership change, a policy launch)
    - instrument = measurement apparatus/equipment
    - material = a physical sample/substance/reagent that must be obtainable
    - method = an analytical/theoretical technique
    - compute = hardware, datasets-for-training, or processing scale
    - access = a means of obtaining data (API tier, archive, registry, permission)
- `hard_or_soft`:
    - "hard" = if absent, the study is impossible (the thing simply did not exist — e.g. an event that had not happened)
    - "soft" = the thing existed but may have been expensive, slow, restricted, or low-scale
- `binding_candidate`: true/false — could this dependency plausibly be the constraint that decides the earliest feasible year? Timeless methods that have existed for decades → false. Recent events, data-access tiers, novel materials, large compute, specialized instruments → usually true. Anything `produced_by_study: true` or any motivational backdrop → false.
- `produced_by_study`: true/false — true if the study CREATES this rather than consuming it (lexicons, datasets, models, features built in the work). Outputs are excluded from feasibility dating.
- `inferred`: true/false (see rule 5).
- `estimated_difficulty`: "basic" | "intermediate" | "advanced"
- `search_query`: an optimized query to locate the *timing origin* of this dependency (see query rules below).

### Search query rules by dependency_type

- **method / instrument**: technical terms, acronyms, key context words. Short precise 3–4 word phrases capturing the core technique or apparatus. Query should help find the earliest literature describing it.
- **data**: locate THIS specific dataset/corpus if it exists publicly. Use the most identifying info — dataset name, institution, domain, language. Template: [dataset_name or corpus_identifier] [domain] [language]
- **event / access / material**: query for when the thing *began to exist or became available* — its origin, announcement, release, or discovery — NOT for a method paper. Template: [thing] [origin/announcement/release] [context]

---

### Example 1 (humanities — concrete requirements)

**Abstract:**
"This thesis is an examination of the national narratives contained in three exhibits in The Museum of New Zealand, Te Papa Tongarewa. It examines the existence of the state and the nation, and their involvement in museum development, and applies this theory, and selected theories of Roland Barthes, Sergei Eisenstein, and Walter Benjamin, to the subsequent analysis. Broadly, the position taken is that museums are one of a number of institutions that perpetuate national narratives in order to bind nations together and discourage anti-state sentiment, and this position is validated in the analysis of three long-term Te Papa exhibits, Exhibiting Ourselves, Parade, and Golden Days."

**Expected Output:**
```json
{{
  "overall_suitable_degree": "MSc",
  "requirements": [
    {{
      "category": "method_technique",
      "name": "National narrative analysis",
      "description": "Examination of national narratives in museum exhibits",
      "evidence_span": "examination of the national narratives contained in three exhibits",
      "dependency_type": "method",
      "hard_or_soft": "soft",
      "binding_candidate": false,
      "produced_by_study": false,
      "inferred": false,
      "estimated_difficulty": "intermediate",
      "search_query": "national narrative analysis museums"
    }},
    {{
      "category": "method_technique",
      "name": "Application of cultural theory",
      "description": "Application of theories from Barthes, Eisenstein, and Benjamin",
      "evidence_span": "selected theories of Roland Barthes, Sergei Eisenstein, and Walter Benjamin",
      "dependency_type": "method",
      "hard_or_soft": "soft",
      "binding_candidate": false,
      "produced_by_study": false,
      "inferred": false,
      "estimated_difficulty": "advanced",
      "search_query": "cultural semiotic theory analysis"
    }},
    {{
      "category": "data",
      "name": "Te Papa Tongarewa exhibits",
      "description": "Analysis of three long-term exhibits as primary material",
      "evidence_span": "three long-term Te Papa exhibits, Exhibiting Ourselves, Parade, and Golden Days",
      "dependency_type": "data",
      "hard_or_soft": "soft",
      "binding_candidate": true,
      "produced_by_study": false,
      "inferred": false,
      "estimated_difficulty": "intermediate",
      "search_query": "Te Papa Tongarewa exhibits New Zealand"
    }}
  ]
}}
```

### Example 2 (empirical — the dependency-extraction move)

**Abstract:**
"Following the change in ownership of Twitter in October 2022, several platform policies changed, particularly content flagging and Twitter Blue verification. We examine shifts in engagement (likes and retweets) for political figures before and after November 2022. We collect tweets from 6550 accounts belonging to political leaders and parties across twelve countries between June 2021 and June 2023, and compare engagement across Left and Right."

**Expected Output:**
```json
{{
  "overall_suitable_degree": "MSc",
  "requirements": [
    {{
      "category": "other",
      "name": "Twitter ownership change",
      "description": "The before/after comparison is defined relative to this event, so it must have occurred",
      "evidence_span": "change in ownership of Twitter in October 2022",
      "dependency_type": "event",
      "hard_or_soft": "hard",
      "binding_candidate": true,
      "produced_by_study": false,
      "inferred": false,
      "estimated_difficulty": "basic",
      "search_query": "Twitter ownership change announcement October 2022"
    }},
    {{
      "category": "data",
      "name": "Bulk tweet collection from 6550 accounts",
      "description": "Engagement metrics for thousands of accounts over two years must be collected at scale",
      "evidence_span": "collect tweets from 6550 accounts",
      "dependency_type": "access",
      "hard_or_soft": "soft",
      "binding_candidate": true,
      "produced_by_study": false,
      "inferred": true,
      "estimated_difficulty": "intermediate",
      "search_query": "Twitter API bulk access tier release"
    }},
    {{
      "category": "method_technique",
      "name": "Before-after engagement comparison",
      "description": "Compares likes and retweets in periods before and after the event",
      "evidence_span": "shifts in engagement (likes and retweets) for political figures before and after November 2022",
      "dependency_type": "method",
      "hard_or_soft": "soft",
      "binding_candidate": false,
      "produced_by_study": false,
      "inferred": false,
      "estimated_difficulty": "basic",
      "search_query": "before after engagement comparison"
    }},
    {{
      "category": "method_technique",
      "name": "Left/Right ideology classification",
      "description": "Accounts are labelled by political ideology for comparison",
      "evidence_span": "compare engagement across Left and Right",
      "dependency_type": "method",
      "hard_or_soft": "soft",
      "binding_candidate": false,
      "produced_by_study": false,
      "inferred": true,
      "estimated_difficulty": "basic",
      "search_query": "political ideology classification left right"
    }}
  ]
}}
```

Note how in Example 2 the *event* and *access* dependencies are the binding candidates (they decide the earliest feasible year), while the *methods* are timeless and marked `binding_candidate: false`. This is the key move: do not stop at naming the technique — surface the event/data/access it sits on top of.

A `produced_by_study: true` example: if the same abstract had said "we construct a new annotated corpus of political tweets", that corpus is an OUTPUT — it would be tagged `"produced_by_study": true`, `"binding_candidate": false`, and excluded from dating, because the study creates it rather than depending on it pre-existing. Contrast with the bulk tweet *collection* above, which depends on a pre-existing access mechanism and so is a real dependency.

---

Now apply the same analysis to the following abstract.

[THESIS ABSTRACT]
{abstract}
[/THESIS ABSTRACT]

Consider dependencies across: datasets/data sources; real-world events or conditions assumed to have occurred; methods/algorithms/theory; tools/libraries/frameworks; compute/hardware; physical samples/materials/instruments; means of data access; and human effort/skills.

### Output Format
Return a valid JSON object with exactly this structure:

```json
{{
  "overall_suitable_degree": "BSc" | "MSc" | "PhD" | "MSc-to-PhD",
  "requirements": [
    {{
      "category": "data" | "method_technique" | "tool_library" | "compute" | "human_effort" | "other",
      "name": "...",
      "description": "...",
      "evidence_span": "verbatim substring from the abstract",
      "dependency_type": "data" | "event" | "instrument" | "material" | "method" | "compute" | "access",
      "hard_or_soft": "hard" | "soft",
      "binding_candidate": true | false,
      "produced_by_study": true | false,
      "inferred": true | false,
      "estimated_difficulty": "basic" | "intermediate" | "advanced",
      "search_query": "..."
    }}
  ]
}}
```

Return only valid JSON. No explanation, markdown fencing, or preamble outside the JSON object.
"""

# Offline fallback: (canonical label, keyword that must appear in the lowercased
# abstract). Ported from the JS `extractMethods` fallback table. First match per
# label wins; capped at 7 to mirror the LLM cap.
KEYWORD_METHODS = [
    ("Large language models", "large language model"),
    ("GPT-4", "gpt-4"),
    ("ChatGPT", "chatgpt"),
    ("LLaMA", "llama"),
    ("LoRA", "lora"),
    ("BERT", "bert"),
    ("RoBERTa", "roberta"),
    ("Transformers", "transformer"),
    ("LSTM", "lstm"),
    ("CNN", "convolutional"),
    ("Word embeddings", "embedding"),
    ("word2vec", "word2vec"),
    ("VADER", "vader"),
    ("SentiWordNet", "sentiwordnet"),
    ("Sentiment lexicons", "lexicon"),
    ("Naïve Bayes", "naive bayes"),
    ("Support Vector Machines", "support vector"),
    ("SVM", "svm"),
    ("Logistic regression", "logistic regression"),
    ("Random forest", "random forest"),
    ("Maximum entropy", "maximum entropy"),
    ("Topic modeling (LDA)", "lda"),
    ("Latent Dirichlet Allocation", "latent dirichlet"),
    ("Twitter data", "twitter"),
    ("Social media data", "social media"),
    ("Ensemble learning", "ensemble"),
    ("Cross-lingual transfer", "cross-lingual"),
    ("Multilingual corpora", "multilingual"),
    ("Aspect-based sentiment", "aspect"),
    ("Fine-tuning", "fine-tun"),
    ("Transfer learning", "transfer learning"),
    ("Variational quantum (VQE/QAOA)", "variational quantum"),
    ("Error correction", "error correction"),
    ("Superconducting qubits", "superconducting qubit"),
    ("Matrix factorization", "matrix factorization"),
    ("Collaborative filtering", "collaborative filtering"),
    ("Federated averaging", "federated averaging"),
    ("Differential privacy", "differential privacy"),
    ("Apache Spark", "spark"),
    ("GPU compute", "gpu"),
    ("Manual annotation", "annotation"),
    ("n-grams", "n-gram"),
    ("TF-IDF", "tf-idf"),
]

_MAX_METHODS = 10


def keyword_extract(text: str) -> List[str]:
    """Offline method extraction: scan the abstract for known method keywords."""
    n = text.lower()
    found: List[str] = []
    for label, kw in KEYWORD_METHODS:
        if kw in n and label not in found:
            found.append(label)
            if len(found) >= _MAX_METHODS:
                break
    return found or ["Sentiment analysis", "Machine learning classifiers"]


def llm_extract(text: str) -> List[str]:
    """LLM method extraction via reusable_parts/tools.get_model().

    Raises on any failure (no key, import error, bad JSON) so the caller can
    fall back to keyword_extract.
    """
    from tools import get_model  # local import: only needed on the LLM path

    llm = get_model()
    if llm is None:
        raise RuntimeError("get_model() returned None (check reusable_parts/config.json)")
    resp = llm.invoke(EXTRACT_PROMPT.format(abstract=text))
    content = getattr(resp, "content", resp)
    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        raise ValueError("no JSON array in LLM response")
    arr = json.loads(match.group(0))
    clean = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    if not clean:
        raise ValueError("LLM returned an empty method list")
    return clean[:_MAX_METHODS]


def extract_methods(text: str, use_llm: bool = True) -> tuple[List[str], str]:
    """Return (methods, how) where how is 'llm' or 'keyword'."""
    if use_llm:
        try:
            return llm_extract(text), "llm"
        except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
            sys.stderr.write(f"[feasibility] LLM extraction unavailable ({exc}); using keywords.\n")
    return keyword_extract(text), "keyword"


# --------------------------------------------------------------------------- #
# 2. Origin-year lookup
# --------------------------------------------------------------------------- #

# Offline estimate table: (keyword, origin_year). Longest matching keyword wins.
# Ported from the JS `localOrigins` table. Used only when the live Semantic
# Scholar lookup returns nothing.
LOCAL_ORIGINS = [
    ("large language model", 2023), ("llm", 2023), ("gpt-4", 2023), ("gpt", 2020),
    ("chatgpt", 2023), ("llama", 2023), ("lora", 2021),
    ("bertopic", 2020), ("roberta", 2019), ("distilbert", 2019), ("bert", 2018),
    ("transformer", 2017), ("attention", 2017),
    ("lstm", 1997), ("gru", 2014), ("cnn", 1998), ("convolutional", 1998),
    ("recurrent neural", 1986), ("deep learning", 2006), ("neural network", 1986),
    ("word2vec", 2013), ("glove", 2014), ("word embedding", 2013), ("fasttext", 2016),
    ("vader", 2014), ("sentiwordnet", 2006), ("afinn", 2011), ("textblob", 2013),
    ("nrc", 2010), ("sentiment lexicon", 2002), ("wordnet", 1995),
    ("naive bayes", 1960), ("naïve bayes", 1960), ("support vector", 1995),
    ("svm", 1995), ("logistic regression", 1958), ("random forest", 2001),
    ("maximum entropy", 1996), ("k-nearest", 1951), ("decision tree", 1986),
    ("latent dirichlet", 2003), ("lda", 2003), ("topic model", 1999),
    ("pagerank", 1998), ("label propagation", 2002), ("graph-based", 2003),
    ("twitter", 2006), ("social media", 2004), ("ensemble", 1990),
    ("gradient boost", 2001), ("xgboost", 2016),
    ("cross-lingual", 2008), ("multilingual", 2007), ("aspect-based", 2010),
    ("fine-tun", 2018), ("transfer learning", 2010), ("data augmentation", 2015),
    ("variational quantum", 2014), ("vqe", 2014), ("qaoa", 2014),
    ("surface code", 1998), ("error correction", 1995), ("error mitigation", 2017),
    ("superconducting qubit", 2011), ("trapped ion", 2003), ("shor", 1994),
    ("grover", 1996), ("logical qubit", 2023),
    ("matrix factorization", 2006), ("collaborative filtering", 1994),
    ("learning to rank", 2005), ("neural recommend", 2016), ("knowledge graph", 2014),
    ("federated averaging", 2016), ("fedavg", 2016), ("secure aggregation", 2017),
    ("differential privacy", 2006),
    ("gpu", 2009), ("apache spark", 2014), ("hadoop", 2006), ("active learning", 1994),
    ("crowdsourc", 2006), ("annotation", 2002), ("pos tag", 1994), ("n-gram", 1948),
    ("tf-idf", 1972), ("bag-of-words", 1954),
]

_DEFAULT_ORIGIN = 2010
_MIN_PLAUSIBLE = 25  # reject an API year more than this many years before the estimate


def guess_origin(name: str) -> int:
    """Offline origin-year estimate: longest-substring match in LOCAL_ORIGINS."""
    n = name.lower()
    best_year, best_len = None, -1
    for kw, yr in LOCAL_ORIGINS:
        if kw in n and len(kw) > best_len:
            best_year, best_len = yr, len(kw)
    return best_year if best_year is not None else _DEFAULT_ORIGIN


def origin_year(name: str, end_year: Optional[int] = None, use_api: bool = True) -> tuple[int, str]:
    """Return (year, source). Tries Semantic Scholar (via method_history) first,
    then falls back to the local estimate. Mirrors the JS fetchOrigin sanity
    check: an API year implausibly earlier than the estimate is rejected."""
    est = guess_origin(name)
    if use_api and _ss_origin_year is not None:
        try:
            # Fail fast: one attempt only. Semantic Scholar rate-limits hard
            # without an API key, and the default 3-retry/5-15s backoff would
            # make a 7-method abstract take minutes. Mirrors the old JS 7s abort.
            yr = _ss_origin_year(name, end_year=end_year, retries=1)
            if yr and 1945 <= yr <= 2026:
                if yr < est - _MIN_PLAUSIBLE:
                    return est, "estimate"
                return yr, "Semantic Scholar"
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[feasibility] origin lookup failed for '{name}' ({exc}); using estimate.\n")
    return est, "estimate"


# --------------------------------------------------------------------------- #
# 3. Orchestration + verdict
# --------------------------------------------------------------------------- #
@dataclass
class MethodResult:
    name: str
    origin: int
    source: str
    anachronistic: bool


@dataclass
class FeasibilityResult:
    year: int
    methods: List[MethodResult]
    extraction: str            # 'llm' or 'keyword'
    anachronistic: bool        # any method postdates `year`
    verdict: str               # human-readable summary

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def analyze(
    abstract: str,
    year: int,
    use_llm: bool = True,
    use_api: bool = True,
) -> FeasibilityResult:
    """Run the full feasibility test for one abstract at a given year."""
    text = (abstract or "").strip()
    if len(text) < 20:
        raise ValueError("Paste an abstract (at least a sentence or two) first.")

    names, how = extract_methods(text, use_llm=use_llm)

    # Look up origins concurrently — live Semantic Scholar calls are I/O-bound,
    # so running them in parallel keeps the whole request responsive instead of
    # summing per-method latency.
    from concurrent.futures import ThreadPoolExecutor

    def _one(nm: str) -> MethodResult:
        yr, src = origin_year(nm, end_year=year, use_api=use_api)
        return MethodResult(name=nm, origin=yr, source=src, anachronistic=yr > year)

    if names:
        with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
            methods = list(ex.map(_one, names))
    else:
        methods = []

    methods.sort(key=lambda m: m.origin)
    late = [m for m in methods if m.anachronistic]
    anachronistic = bool(late)
    if anachronistic:
        names_late = ", ".join(f"{m.name} ({m.origin})" for m in late)
        verdict = (
            f"Anachronistic for {year}: {len(late)} method"
            f"{'s' if len(late) > 1 else ''} postdate{'s' if len(late) == 1 else ''} it — {names_late}."
        )
    else:
        verdict = f"Plausible for {year}: all {len(methods)} detected methods existed by then."

    return FeasibilityResult(
        year=year,
        methods=methods,
        extraction=how,
        anachronistic=anachronistic,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Test an abstract for feasibility at a given year.")
    p.add_argument("--abstract", help="Path to a file with the abstract text. Omit to read stdin.")
    p.add_argument("--year", type=int, required=True, help="Year to test the abstract against.")
    p.add_argument("--no-llm", action="store_true", help="Skip the LLM; use keyword extraction only.")
    p.add_argument("--no-api", action="store_true", help="Skip Semantic Scholar; use local estimates only.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = p.parse_args(argv)

    if args.abstract:
        with open(args.abstract, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    try:
        res = analyze(text, args.year, use_llm=not args.no_llm, use_api=not args.no_api)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    print(f"Detected methods ({len(res.methods)})  [extraction: {res.extraction}]:")
    for m in res.methods:
        flag = "⚠ ANACHRONISTIC" if m.anachronistic else "OK"
        print(f"  {m.name:34s} origin {m.origin}  [{m.source}]  {flag}")
    print(f"\nVerdict: {res.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
