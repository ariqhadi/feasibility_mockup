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
EXTRACT_PROMPT = """You are an experienced Computer Science professor extracting the research \
methods from a thesis or grant abstract.

List the distinct methods, models, algorithms, and data sources actually used or \
proposed in the abstract. Be strict and conservative — do NOT invent methods that \
are not supported by the text. Prefer short, canonical names (acronyms are fine), \
most specific first.

Return ONLY a compact JSON array of up to 7 strings. No prose, no markdown fencing.
Example: ["BERT", "Twitter data", "Topic modeling (LDA)"]

[ABSTRACT]
{abstract}
[/ABSTRACT]
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
