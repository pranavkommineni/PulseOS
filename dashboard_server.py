"""
PulseOS Dashboard Server
=========================

Bridges the three CSV files produced by `integration/live_pipeline.py`
(Person 1 / Person 2 / Person 3 metrics) to the live web dashboard in
`dashboard/index.html`, over a single WebSocket connection.

It does NOT re-implement any of the health-intelligence / RSUL logic. It
only *watches* the CSV files that the pipeline already writes to disk and
streams each new row-triplet to every connected browser the moment it
appears, formatted exactly the way the dashboard's `render(msg)` function
expects:

    {
      "type": "sample",
      "part1": {...raw firmware/telemetry columns...},
      "part2": {...health-intelligence columns...},
      "part3": {...rsul-control columns...},
      "meta":  {"source": "csv", "detail": "...", "scenario_label": "...",
                "samples_processed": N}
    }

Run this in a SEPARATE terminal from the pipeline itself:

    # Terminal 1 — generates the CSVs (demo/replay/live, see README)
    python integration/live_pipeline.py --demo

    # Terminal 2 — serves the dashboard + streams the CSVs to it
    python dashboard_server.py

Then open http://localhost:8000 in a browser.

Both processes must be run from the PulseOS-main project root, since the
pipeline writes amr_dataset.csv / health_intelligence_output.csv /
rsul_control_output.csv relative to the current working directory.
"""

import asyncio
import csv
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_HTML = PROJECT_ROOT / "dashboard" / "index.html"

RAW_CSV_PATH = PROJECT_ROOT / "amr_dataset.csv"
HEALTH_CSV_PATH = PROJECT_ROOT / "health_intelligence_output.csv"
RSUL_CSV_PATH = PROJECT_ROOT / "rsul_control_output.csv"

# How often to check the CSV files for newly appended rows.
POLL_SECONDS = 0.3


# ============================================================================
# INCREMENTAL CSV TAILER
# ============================================================================


class CsvTailer:
    """
    Reads only the rows appended to a CSV file since the last check.

    Safe to use against a file another process is actively appending to,
    because live_pipeline.py writes each row atomically (open in append
    mode, write exactly one row, close) before moving on to the next
    reading.
    """

    def __init__(self, path: Path):
        self.path = path
        self.header: Optional[List[str]] = None
        self.offset = 0

    def read_new_rows(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []

        with open(self.path, "r", newline="") as f:
            if self.header is None:
                header_line = f.readline()
                if not header_line:
                    return []  # file exists but header not written yet
                self.header = next(csv.reader(io.StringIO(header_line)))
                self.offset = f.tell()

            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()

        if not chunk.strip():
            return []

        rows = []
        for raw_row in csv.reader(io.StringIO(chunk)):
            if raw_row:
                rows.append(dict(zip(self.header, raw_row)))
        return rows


# ============================================================================
# APP STATE
# ============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not DASHBOARD_HTML.exists():
        raise RuntimeError(f"Dashboard HTML not found at {DASHBOARD_HTML}")
    task = asyncio.create_task(poll_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="PulseOS Dashboard Server", lifespan=lifespan)

clients: "set[WebSocket]" = set()

raw_tailer = CsvTailer(RAW_CSV_PATH)
health_tailer = CsvTailer(HEALTH_CSV_PATH)
rsul_tailer = CsvTailer(RSUL_CSV_PATH)

# Rows that have been read from a faster-moving CSV but are still waiting
# for their matching row to show up in the other two files.
pending_raw: List[Dict[str, str]] = []
pending_health: List[Dict[str, str]] = []
pending_rsul: List[Dict[str, str]] = []

samples_processed = 0
pipeline_seen = False


# ============================================================================
# BROADCAST
# ============================================================================


async def broadcast(message: dict) -> None:
    if not clients:
        return
    data = json.dumps(message, default=str)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


# ============================================================================
# POLL LOOP — watches the 3 CSVs and emits matched row-triplets
# ============================================================================


async def poll_loop():
    global samples_processed, pipeline_seen

    while True:
        pending_raw.extend(raw_tailer.read_new_rows())
        pending_health.extend(health_tailer.read_new_rows())
        pending_rsul.extend(rsul_tailer.read_new_rows())

        if not pipeline_seen and (pending_raw or pending_health or pending_rsul):
            pipeline_seen = True
            await broadcast(
                {
                    "type": "status",
                    "meta": {
                        "source": "csv",
                        "detail": "pipeline output detected — streaming",
                        "status": "streaming",
                    },
                }
            )

        while pending_raw and pending_health and pending_rsul:
            p1 = pending_raw.pop(0)
            p2 = pending_health.pop(0)
            p3 = pending_rsul.pop(0)

            p1 = {k: v for k, v in p1.items() if k != "logged_at"}

            samples_processed += 1

            await broadcast(
                {
                    "type": "sample",
                    "part1": p1,
                    "part2": p2,
                    "part3": p3,
                    "meta": {
                        "source": "csv",
                        "detail": f"{RAW_CSV_PATH.name} / {HEALTH_CSV_PATH.name} / {RSUL_CSV_PATH.name}",
                        "scenario_label": p1.get("scenario_id"),
                        "samples_processed": samples_processed,
                    },
                }
            )

        await asyncio.sleep(POLL_SECONDS)


# ============================================================================
# ROUTES
# ============================================================================


@app.get("/")
async def index():
    return FileResponse(DASHBOARD_HTML)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    await websocket.send_text(
        json.dumps(
            {
                "type": "status",
                "meta": {
                    "source": "csv",
                    "detail": (
                        "connected — waiting for "
                        f"{RAW_CSV_PATH.name} to receive data"
                        if not pipeline_seen
                        else "reconnected — streaming"
                    ),
                    "status": "connected",
                },
            }
        )
    )

    try:
        while True:
            # The dashboard doesn't send anything; this just keeps the
            # connection open and detects client disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    print("PulseOS dashboard: http://localhost:8000")
    print(f"Watching for CSVs in: {PROJECT_ROOT}")
    print("  -", RAW_CSV_PATH.name)
    print("  -", HEALTH_CSV_PATH.name)
    print("  -", RSUL_CSV_PATH.name)
    print(
        "If those files don't exist yet, run the pipeline in another "
        "terminal first, e.g.:\n"
        "  python integration/live_pipeline.py --demo"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
