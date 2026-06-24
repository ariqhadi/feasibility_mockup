# Field Shift Explorer

Interactive view of how a research field's method stack turns over, year by year
(2003–2026), plus a feasibility view (when methods get funded vs. attempted in
theses) and an abstract-anachronism tester.

Imported from the Claude Design project and wired to the **real** data in `data/`.

## Layout

```
mockup/
├── index.html                    # redirect → the Design Component page
├── Field Shift Explorer v2.dc.html  # the app (template + React logic)
├── support.js                    # Design Component runtime (loads React, boots the page)
├── field-data.js                 # GENERATED — the data the app renders
├── build_field_data.py           # regenerates field-data.js from data/
└── data/
    ├── sentiment_analysis/
    │   ├── raw_data.csv
    │   ├── per_year_summaries.json
    │   └── all_year_summaries.json
    └── superconductor/
        └── … (same three files)
```

## The data is dynamic

The web shows **whatever field folders live under `data/`**. Each field is a
folder containing exactly three files:

- `raw_data.csv` — one row per document (`;`-separated); columns include
  `data_id`, `Academic Level` (`grant` / `PhD` / `MA`), `Published Year`.
- `per_year_summaries.json` — `{ "<year>": { summary, requirements:[{category,name,…}] } }`.
- `all_year_summaries.json` — includes `turning_points:[{period,change,…}]`.

`build_field_data.py` derives everything the app renders from these files:

- **PhD / MSc / grant** lines = real distinct-document counts per year.
- **Per-year story + requirement chips** = the per-year summaries (chips grouped
  into Data / Methods / Tools / Compute / Human).
- **Turning points → eras** and the **method / data / compute "lifelines"**
  (which drive the coloured "method enters the field" dots and the feasibility
  lag plot), plus a maturity estimate and tagline — all derived deterministically.

To change what the web shows: **add or edit a folder under `data/`, then run**

```bash
python3 build_field_data.py
```

(`sentiment_analysis` is listed first when present; otherwise fields are
alphabetical.)

## Run it

A static file server is enough (the runtime fetches `support.js` and
`field-data.js` relatively, and reads no data files at runtime):

```bash
cd mockup
python3 -m http.server 8000
# open http://localhost:8000/
```
