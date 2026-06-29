"""
server.py
---------
Local backend for the Field Shift Explorer's "Test an abstract for feasibility"
feature. It does two things:

  1. Serves the static site (so the page and the API share one origin — no CORS).
  2. Exposes POST /analyze, which runs feasibility.analyze() and returns JSON the
     page renders directly.

Run:
    python server.py            # http://localhost:8000/
    python server.py --port 5001

Then open http://localhost:8000/ and use "Test an abstract".

The analyze logic lives in feasibility.py — edit that, not this file, to change
how methods are extracted or dated. This file is just transport.
"""

from __future__ import annotations

import argparse
import os

from flask import Flask, jsonify, request, send_from_directory

import feasibility

_HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = "Field Shift Explorer v2.dc.html"

app = Flask(__name__, static_folder=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    """{abstract, year, use_llm?, use_api?} -> feasibility result JSON."""
    body = request.get_json(force=True, silent=True) or {}
    abstract = body.get("abstract", "")
    try:
        year = int(body.get("year"))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid integer 'year' is required."}), 400

    # Defaults match the CLI: try LLM + live API, fall back gracefully. The page
    # can override per request (e.g. {"use_api": false} for instant offline runs).
    # use_llm = bool(body.get("use_llm", True))
    # use_api = bool(body.get("use_api", True))

    try:
        result = feasibility.analyze(abstract, year)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - never 500 silently; surface to the UI
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    return jsonify(result.to_dict())


@app.route("/")
def index():
    return send_from_directory(_HERE, PAGE)


@app.route("/<path:path>")
def static_files(path):
    """Serve field-data.js, support.js, and anything else the page references."""
    return send_from_directory(_HERE, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the Field Shift Explorer + /analyze API.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    print(f"Field Shift Explorer + feasibility API on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
