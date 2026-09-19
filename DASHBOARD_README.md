# PulseOS Live Dashboard — Setup & Run Guide

This adds a live web dashboard on top of the existing PulseOS pipeline.
It does **not** change any of the health-intelligence / rsul-control logic —
it only *displays* what the pipeline already produces.

## What was added

```
PulseOS-main/
├── dashboard/
│   └── index.html          <- the frontend you supplied (unmodified logic)
├── dashboard_server.py      <- NEW: watches the pipeline's CSVs and streams
│                                them to the browser over a WebSocket
└── dashboard_requirements.txt  <- NEW: fastapi + uvicorn
```

**One bug was fixed** in `integration/live_pipeline.py`: `main()` referenced
the module-level `TREND_WINDOW` global before its `global` declaration,
which raised `SyntaxError: name 'TREND_WINDOW' is used prior to global
declaration` and prevented the whole pipeline (demo/replay/live) from
running at all. Fixed by moving the `global TREND_WINDOW` statement to the
top of `main()`.

## How it fits together

```
data_collection / ESP32          health-intelligence         rsul-control
(Person 1 metrics)      ──►     (Person 2 metrics)   ──►    (Person 3 metrics)
        │                                │                          │
        ▼                                ▼                          ▼
  amr_dataset.csv        health_intelligence_output.csv    rsul_control_output.csv
        │                                │                          │
        └────────────────────────────────┴──────────────────────────┘
                                     │
                          dashboard_server.py (tails all 3 CSVs,
                          matches rows, pushes over WebSocket)
                                     │
                                     ▼
                          dashboard/index.html (your frontend)
```

`integration/live_pipeline.py` already writes exactly these three CSVs, one
matched row per reading:

- `amr_dataset.csv` — raw Person‑1 telemetry (cpu, heap, stack, tasks, AI
  inference, queue, messaging, power, resets, …)
- `health_intelligence_output.csv` — Person‑2 derived health metrics
  (health scores, trends, fault flags, degradation index, …)
- `rsul_control_output.csv` — Person‑3 predictive output (RSUL hours,
  failure probability, risk level, recommendation, …)

`dashboard_server.py` is new glue: it watches all three files, waits until
a matching row has landed in each, and pushes it to every connected
browser as one JSON message shaped exactly the way `dashboard/index.html`
expects (`{type:"sample", part1, part2, part3, meta}`). This keeps the
dashboard fully decoupled from *how* the CSVs are produced — live ESP32
hardware, `--replay` of a saved CSV, or `--demo` synthetic data all work
identically, because they all go through the same three CSV files.

## Setup

```bash
cd PulseOS-main

# Python deps for the health-intelligence + rsul-control pipeline
pip install -r health-intelligence/requirements.txt
pip install -r rsul-control/requirements.txt

# Only needed if you'll capture from real ESP32-S3 hardware
pip install pyserial openpyxl

# Python deps for the new dashboard server
pip install -r dashboard_requirements.txt
```

(Use a virtualenv if you prefer: `python3 -m venv .venv && source .venv/bin/activate` first.)

## Run it

You need **two terminals**, both opened at the `PulseOS-main/` project
root (the CSVs are written/read relative to the current directory).

**Terminal 1 — generate live data.** Pick ONE of these:

```bash
# Option A: no hardware needed — synthetic telemetry through the real
# health-intelligence + rsul-control engines (best for testing this setup)
python integration/live_pipeline.py --demo

# Option B: replay a previously captured CSV through the pipeline
python integration/live_pipeline.py --replay amr_dataset.csv

# Option C: real ESP32-S3 hardware flashed with arduino_ide.ino, over USB
python integration/live_pipeline.py --load normal
```

Each of these appends matching rows to `amr_dataset.csv`,
`health_intelligence_output.csv`, and `rsul_control_output.csv` in the
project root as it runs.

**Terminal 2 — serve the dashboard:**

```bash
python dashboard_server.py
```

Then open **http://localhost:8000** in your browser. The dashboard will
show `CONNECTING` until the first matched row is available, then flip to
`LIVE` and start updating in real time as new rows are appended.

## Notes

- Restart `integration/live_pipeline.py` fresh runs will keep *appending*
  to the same three CSVs. Delete `amr_dataset.csv`,
  `health_intelligence_output.csv`, and `rsul_control_output.csv` before a
  clean run if you don't want old rows replayed into the dashboard on
  startup (the server only streams rows appended *after* it starts
  watching — but any leftover in the file from a previous run is
  automatically included first).
- `dashboard_server.py` never modifies the CSVs; it's read-only.
- If port 8000 is taken, edit the `uvicorn.run(..., port=8000)` line at
  the bottom of `dashboard_server.py`.
