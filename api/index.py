"""
api/index.py
------------
Vercel entrypoint for the feasibility backend.

Vercel only turns files inside this `api/` directory into Serverless Functions,
so this thin module re-exports the Flask `app` defined in the project-root
`server.py`. The real transport + analysis logic still lives in `server.py` /
`feasibility.py` — edit those, not this file.

Only POST /analyze is routed here (see vercel.json `rewrites`); the static page
and its assets are served directly by Vercel's static layer.
"""

import os
import sys

# server.py and feasibility.py live one level up; put the project root on the
# import path so `from server import app` resolves both here and in the bundle.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server import app  # noqa: E402  (path setup must precede the import)

# Vercel's Python runtime detects this top-level `app` (WSGI) automatically.
__all__ = ["app"]
