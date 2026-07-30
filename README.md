# PulseOS

**Predictive runtime software health, RSUL estimation, and self-adaptive resource management for RTOS-based Edge AI systems.**

PulseOS gives an embedded Edge AI device a pulse: it continuously senses its own runtime health, understands *why* that health is changing, predicts how much reliable operating time is left, and automatically adapts itself to extend that time — without breaking the application's own performance requirements.

---
## Folder Architecture
software-health-rsul-framework/
├── README.md
├── .gitignore
│
├── docs/
│   ├── project_draft/                  # the proposal doc, diagrams
│   ├── architecture/
│   │   └── system_architecture.md
│   ├── interfaces/                     # <- single source of truth, read this first
│   │   ├── person1_to_person2.schema.json
│   │   ├── person2_to_person3.schema.json
│   │   ├── person3_to_person1_action.schema.json
│   │   └── CHANGELOG.md                # version every schema change
│   └── experiments/
│       ├── protocol.md
│       └── results/
│
├── firmware/                            # PERSON 1
│   ├── CMakeLists.txt / sdkconfig       # ESP-IDF project
│   ├── main/
│   │   ├── tasks/                      # sensor, inference, comm, monitoring, health_manager
│   │   ├── context/                    # context_detector.c/.h
│   │   ├── fingerprint/                # behavioral_fingerprint.c/.h
│   │   ├── degradation/                # controlled degradation generator
│   │   ├── comms/                      # mqtt_client.c, uart_bridge.c
│   │   └── model/                      # tflite model + data
│   ├── tools/
│   │   └── log_to_json.py              # serial log -> schema-conformant JSON
│   └── tests/
│
├── health-intelligence/                 # PERSON 2
│   ├── requirements.txt
│   ├── src/health_intel/
│   │   ├── ingestion/                  # reads Person 1's stream
│   │   ├── preprocessing/              # cleaning, normalization, feature extraction
│   │   ├── shi/                        # context_weighting.py, shi_model.py
│   │   ├── trend/                      # trend_analysis.py
│   │   ├── root_cause/                 # attribution_model.py
│   │   ├── classification/             # health_state.py
│   │   └── api/                        # publish_to_person3.py
│   ├── notebooks/                      # 01_eda, 02_shi_dev, 03_root_cause_dev
│   ├── models/                         # saved joblib artifacts
│   ├── data/{raw,processed,synthetic}/
│   └── tests/
│
├── rsul-control/                        # PERSON 3
│   ├── requirements.txt
│   ├── src/rsul_control/
│   │   ├── rsul/                       # rsul_model.py, uncertainty.py
│   │   ├── thresholds/                 # failure_definitions.py
│   │   ├── actions/                    # candidate_generator.py, outcome_predictor.py
│   │   ├── optimization/                # constrained_optimizer.py
│   │   ├── controller/                 # action_selector.py, rtos_bridge.py
│   │   └── api/                        # FastAPI service (main.py)
│   ├── dashboard/                       # streamlit_app.py
│   ├── models/
│   ├── data/
│   └── tests/
│
├── integration/                         # SHARED — the closed-loop glue
│   ├── mqtt_broker/docker-compose.yml
│   ├── db/schema.sql                    # shared SQLite schema (Person 2 & 3)
│   ├── e2e_pipeline/run_full_loop.py    # runs the whole monitor→adapt→observe cycle
│   └── ci/github-actions/
│
└── scripts/
    ├── setup_env.sh
    ├── flash_firmware.sh
    └── generate_synthetic_dataset.py
---


## Table of contents

- [PulseOS](#pulseos)
  - [Folder Architecture](#folder-architecture)
  - [Table of contents](#table-of-contents)
  - [Why PulseOS](#why-pulseos)
  - [Core concept](#core-concept)
  - [Architecture](#architecture)
  - [Repository structure](#repository-structure)
  - [Data contracts](#data-contracts)
  - [Getting started](#getting-started)
  - [Tech stack](#tech-stack)
  - [Roadmap](#roadmap)
  - [Evaluation](#evaluation)
  - [Team](#team)
  - [Research \& patent notes](#research--patent-notes)
  - [License](#license)

---

## Why PulseOS

Most embedded monitoring stops at a dashboard: CPU, memory, a few graphs. That tells you what already happened. PulseOS instead:

- Learns what *normal* looks like for whatever the device is currently doing, so a busy-but-healthy device isn't confused with a failing one.
- Converts many raw metrics into one dynamic, context-aware health score.
- Explains *why* health is dropping, not just that it is.
- Predicts remaining useful operating time, with an honest uncertainty range.
- Tests candidate fixes by predicting their outcome *before* applying them.
- Picks the fix that best extends runtime without violating accuracy, latency, memory, or energy constraints.
- Applies it, watches what actually happens, and updates itself.

## Core concept

```
Monitor
   ↓
Understand operating context
   ↓
Learn behavioral fingerprint
   ↓
Calculate dynamic software health (SHI)
   ↓
Identify cause of degradation
   ↓
Predict RSUL + uncertainty
   ↓
Simulate possible recovery actions
   ↓
Select best action
   ↓
Adapt RTOS
   ↓
Observe result
   └──────────────► feedback loop
```

## Architecture

PulseOS is built as three cooperating subsystems, each with a clear question to answer:

| Subsystem                                     | Question it answers                                                                            | Owner    |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------- |
| [`firmware/`](firmware)                       | What is happening inside the device, and is its behavior normal for what it's currently doing? | Person 1 |
| [`health-intelligence/`](health-intelligence) | How healthy is the software, how is that health changing, and what's causing the degradation?  | Person 2 |
| [`rsul-control/`](rsul-control)               | How much reliable operation remains, and what should the system do now to extend it?           | Person 3 |

Data flows one way around the loop, and the selected action flows back to close it:

```
firmware/  ──runtime features──►  health-intelligence/  ──health intelligence──►  rsul-control/
   ▲                                                                                    │
   └───────────────────────────── selected action ────────────────────────────────────┘
```

The adaptive controller at the end is intentionally shared: `rsul-control` decides *what* to do, `firmware` implements *how* the RTOS does it.

## Repository structure

```
pulseos/
├── docs/
│   ├── architecture/          # system diagrams, design notes
│   ├── interfaces/            # versioned JSON schemas — read this first
│   └── experiments/           # protocol + results
├── firmware/                  # Person 1 — ESP32-S3 / FreeRTOS / ESP-IDF
├── health-intelligence/       # Person 2 — Python health scoring & root cause
├── rsul-control/              # Person 3 — Python RSUL prediction & optimization
├── integration/               # shared glue: MQTT broker, DB schema, e2e loop runner
└── scripts/                   # setup, flashing, synthetic data generation
```

See [`docs/architecture/system_architecture.md`](docs/architecture/system_architecture.md) for the full breakdown of every module inside each subsystem.

## Data contracts

Every handoff between subsystems is a versioned JSON schema in [`docs/interfaces/`](docs/interfaces). Change a field, bump the schema, log it in `CHANGELOG.md`.

**firmware → health-intelligence**
```json
{
  "timestamp": 220,
  "context": "AI_INTENSIVE",
  "cpu": 82,
  "heap": 72000,
  "stack": 1840,
  "inference_ms": 131,
  "queue_latency": 21,
  "deadline_miss_rate": 0.11,
  "behavior_deviation": 0.67
}
```

**health-intelligence → rsul-control**
```json
{
  "timestamp": 220,
  "shi": 57,
  "health_state": "DEGRADING",
  "degradation_rate": -3.4,
  "degradation_acceleration": 0.8,
  "root_cause": "MEMORY_PRESSURE",
  "cause_confidence": 0.81
}
```

**rsul-control → firmware**
```json
{
  "action": "SWITCH_MODEL",
  "model": "LIGHTWEIGHT",
  "reason": "AI_OVERLOAD",
  "predicted_rsul_gain": 21
}
```

## Getting started

```bash
git clone https://github.com/<org>/pulseos.git
cd pulseos
./scripts/setup_env.sh          # installs Python deps for both ML subsystems
```

**Firmware (Person 1)**
```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

**Health intelligence (Person 2)**
```bash
cd health-intelligence
pip install -r requirements.txt
python -m health_intel.ingestion.loader --input ../data/synthetic
```

**RSUL control (Person 3)**
```bash
cd rsul-control
pip install -r requirements.txt
uvicorn rsul_control.api.main:app --reload
streamlit run dashboard/streamlit_app.py
```

**Full closed loop (simulation, no hardware required)**
```bash
python integration/e2e_pipeline/run_full_loop.py --scenario degrading
```

## Tech stack

| Layer               | Stack                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------- |
| Firmware            | C/C++, FreeRTOS, ESP-IDF, ESP32-S3, TensorFlow Lite Micro, MQTT, UART                 |
| Health intelligence | Python, Pandas, NumPy, Scikit-learn, XGBoost, Jupyter, SQLite                         |
| RSUL control        | Python, Scikit-learn/XGBoost, PyTorch/TensorFlow (optional), FastAPI, Streamlit, MQTT |

## Roadmap

| Weeks | Phase           | Gate                                              |
| ----- | --------------- | ------------------------------------------------- |
| 1–2   | Foundation      | Shared JSON/CSV contract locked                   |
| 3–4   | Core modules    | `firmware → SHI → RSUL` chain runs end to end     |
| 5–6   | Intelligence    | Root cause + candidate actions added              |
| 7–8   | Adaptive system | Predictions drive real RTOS actions (loop closed) |
| 9–10  | Experiments     | Baseline vs. proposed system measured             |

## Evaluation

Baseline (no prediction, no adaptation) and proposed (full PulseOS loop) are run under identical induced-degradation scenarios and compared on: RSUL prediction error, false-alarm rate, SHI quality, threshold-crossing time extension, deadline misses, inference latency, AI accuracy, CPU/RAM/energy overhead, and recovery success rate. See [`docs/experiments/protocol.md`](docs/experiments/protocol.md).

## Team

|       | Person 1        | Person 2               | Person 3             |
| ----- | --------------- | ---------------------- | -------------------- |
| Owns  | `firmware/`     | `health-intelligence/` | `rsul-control/`      |
| Focus | Embedded / RTOS | Health intelligence    | Prediction / control |

## Research & patent notes

The individual pieces (ML-based health scoring, RSUL prediction) are not novel in isolation. The claimed contribution is the specific combination: context-aware behavioral fingerprinting + dynamic multi-metric health estimation + root-cause-aware RSUL prediction with uncertainty + predicted-outcome candidate actions + constraint-aware closed-loop RTOS adaptation. A prior-art search is a required step before any patent filing.

## License

TBD — add a `LICENSE` file before making this repository public.