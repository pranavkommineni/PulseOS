#!/usr/bin/env python3
"""
PulseOS dashboard demo server
=============================

Serves the PulseOS dashboard HTML and streams live samples to it over the
WebSocket endpoint the dashboard already expects (``/ws``), using the same
moderate ESP32-S3 telemetry generator as the terminal demo.

    python demo_dashboard.py                       # 5 min run, real time
    python demo_dashboard.py --minutes 10 --speed 0.25
    python demo_dashboard.py --html dashboard.html --port 8080
    python demo_dashboard.py --loop                # restart when the run ends

Each sample is pushed as:

    {"type": "sample",
     "part1": {...raw telemetry...},
     "part2": {...health intelligence...},
     "part3": {...rsul prediction...},
     "meta":  {...source / scenario / progress...}}

Pipeline: the sibling script defining ``TrendTracker`` and ``process_reading``
is imported automatically (same rule as the terminal demo). If it cannot be
found, a built-in fallback health + RSUL engine is used so the dashboard still
runs; pass --require-pipeline to make a missing pipeline a hard error instead.

No third-party packages required — HTTP and WebSocket are implemented on the
standard library.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import sys
import threading
import time
import webbrowser
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ============================================================================
# LOAD YOUR PIPELINE MODULE
# ============================================================================


def load_pipeline(explicit=None, required=False):
    """Import the sibling script that defines TrendTracker + process_reading."""
    if explicit:
        candidates = [Path(explicit).resolve()]
    else:
        me = Path(__file__).resolve()
        candidates = [p for p in sorted(HERE.glob("*.py")) if p.resolve() != me]

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "class TrendTracker" in text and "def process_reading" in text:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            return module

    if required:
        sys.exit(
            "Could not find the pipeline script (the file that defines TrendTracker "
            "and process_reading). Put it in the same folder, or pass "
            "--pipeline path/to/that_script.py"
        )
    return None


# ============================================================================
# MODERATE ESP32-S3 TELEMETRY GENERATOR
# ============================================================================


class ModerateEsp32S3:
    """
    Produces one 48-column reading per call (same keys as the firmware header).

    Design rules so the health engine sees a *moderate but healthy* device:
      - heap / stack are smooth (no steps or spikes) so no false leak / stack-risk trend
      - load bursts touch CPU, scheduler delay, inference time and queue only
      - counters are cumulative and monotonic, min/avg/max are consistent
    """

    TOTAL_HEAP = 327_680  # ~320 KB usable internal heap on an S3
    STACK_BYTES = 8_192
    QUEUE_CAP = 64
    PERIOD_MS = 50.0
    INFER_HZ = 10.0

    # Fractions of the run at which things happen (edit freely).
    BURST_AT = (0.22, 0.48, 0.75)  # inference-batch load bursts
    MISS_AT = (0.55, 0.85)  # deadline misses
    DROP_AT = (0.68,)  # dropped messages
    TIMEOUT_AT = (0.40,)  # ultrasonic timeouts

    def __init__(self, total_seconds, step=1.0, seed=7, start_time=1000.0):
        self.rng = random.Random(seed)
        self.T = max(float(total_seconds), 1.0)
        self.step = float(step)
        self.start_time = start_time
        self.i = 0

        self.min_free = self.TOTAL_HEAP
        self.lat_ema = None
        self.lat_win = deque(maxlen=20)
        self.dist = 95.0
        self.c = {
            "exec": 0,
            "infer": 0,
            "ctx": 0,
            "task_sw": 0,
            "sent": 0,
            "recv": 0,
            "dropped": 0,
            "irq": 0,
            "misses": 0,
            "timeouts": 0,
        }

        def idx(fracs):
            return {max(1, int(round(f * self.T / self.step))) for f in fracs}

        self.ev_miss = idx(self.MISS_AT)
        self.ev_drop = idx(self.DROP_AT)
        self.ev_timeout = idx(self.TIMEOUT_AT)

    def next_reading(self):
        rng, i, step, c = self.rng, self.i, self.step, self.c
        t = i * step
        p = min(1.0, t / self.T)
        burst = min(
            1.0, sum(math.exp(-(((t - f * self.T) / 4.0) ** 2)) for f in self.BURST_AT)
        )

        # ---- CPU ------------------------------------------------------------
        cpu = 46.0 + 5.0 * p + 12.0 * burst + rng.gauss(0, 1.6)
        cpu = round(max(25.0, min(78.0, cpu)), 2)
        idle = round(100.0 - cpu, 2)
        task_cpu = round(cpu * 0.55, 2)

        # ---- Heap (steady, tiny drift, no burst coupling) ---------------------
        used = int(self.TOTAL_HEAP * (0.52 + 0.02 * p + rng.gauss(0, 0.0006)))
        free = self.TOTAL_HEAP - used
        self.min_free = min(self.min_free, free - int(rng.uniform(1500, 4000)))
        heap_util = round(used / self.TOTAL_HEAP * 100.0, 2)

        # ---- Stack (steady) ---------------------------------------------------
        stack_util = round(51.5 + 0.8 * p + rng.gauss(0, 0.05), 2)
        stack_hwm = int(self.STACK_BYTES * (1.0 - stack_util / 100.0))

        # ---- Task timing --------------------------------------------------------
        exec_ms = round(20.5 + 0.05 * (cpu - 46.0) + rng.gauss(0, 0.5), 2)
        jitter = round(1.6 + 1.2 * burst + abs(rng.gauss(0, 0.5)), 2)
        sched = round(3.6 + 3.0 * burst + abs(rng.gauss(0, 0.6)), 2)
        if i in self.ev_miss:
            c["misses"] += 1

        # ---- AI inference -------------------------------------------------------
        inf = round(max(30.0, 39.5 + 3.5 * p + 5.0 * burst + rng.gauss(0, 0.9)), 2)
        self.lat_ema = inf if self.lat_ema is None else 0.8 * self.lat_ema + 0.2 * inf
        self.lat_win.append(inf)
        avg = round(self.lat_ema, 2)
        lo = round(min(min(self.lat_win), avg), 2)
        hi = round(max(max(self.lat_win), avg), 2)

        # ---- Queue ----------------------------------------------------------------
        q_len = int(round(20 + 4 * burst + rng.gauss(0, 0.8)))
        q_len = max(2, min(self.QUEUE_CAP - 1, q_len))
        q_util = round(q_len / self.QUEUE_CAP * 100.0, 2)

        # ---- Cumulative counters --------------------------------------------------
        c["exec"] += int(step * 1000 / self.PERIOD_MS)
        c["infer"] += int(step * self.INFER_HZ)
        c["ctx"] += int(rng.gauss(950, 40) * step)
        c["task_sw"] += int(rng.gauss(620, 30) * step)
        sent = max(0, int(rng.gauss(15, 2) * step))
        c["sent"] += sent
        c["recv"] += max(0, sent - (1 if rng.random() < 0.05 else 0))
        if i in self.ev_drop:
            c["dropped"] += 1
        if i in self.ev_timeout:
            c["timeouts"] += 1
        c["irq"] += int(rng.gauss(140, 12) * step)

        # ---- Power / thermal / misc ---------------------------------------------------
        power = round(780.0 + 4.5 * cpu + rng.gauss(0, 15), 1)
        temp = round(48.0 + 5.0 * p + 0.06 * (cpu - 46.0) + rng.gauss(0, 0.25), 1)
        irq_lat = round(max(0.1, rng.gauss(0.45, 0.06)), 3)
        self.dist = max(30.0, min(150.0, self.dist + rng.gauss(0, 2.5)))
        churn = int(max(0, rng.gauss(14000, 2500) + 6000 * burst))

        reading = {
            "timestamp": round(self.start_time + t, 3),
            "sample_id": i + 1,
            "uptime_ms": round(60_000.0 + t * 1000.0, 1),
            "scenario_id": "1",
            "cpu_utilization": cpu,
            "cpu_idle": idle,
            "task_cpu_utilization": task_cpu,
            "total_heap": self.TOTAL_HEAP,
            "free_heap": free,
            "used_heap": used,
            "heap_utilization": heap_util,
            "minimum_free_heap": self.min_free,
            "stack_high_water_mark": stack_hwm,
            "stack_utilization": stack_util,
            "task_name": "ai_inference_task",
            "task_priority": 5,
            "task_state": "RUNNING",
            "task_execution_time": exec_ms,
            "task_period": self.PERIOD_MS,
            "task_jitter": jitter,
            "task_execution_count": c["exec"],
            "deadline_misses": c["misses"],
            "context_switches": c["ctx"],
            "task_switches": c["task_sw"],
            "scheduler_delay": sched,
            "active_task_count": 9,
            "inference_time": inf,
            "min_inference_time": lo,
            "max_inference_time": hi,
            "average_inference_time": avg,
            "inference_count": c["infer"],
            "inference_frequency": self.INFER_HZ,
            "queue_length": q_len,
            "queue_capacity": self.QUEUE_CAP,
            "queue_utilization": q_util,
            "messages_sent": c["sent"],
            "messages_received": c["recv"],
            "dropped_messages": c["dropped"],
            "interrupt_count": c["irq"],
            "interrupt_latency": irq_lat,
            "power_consumption": power,
            "system_temperature": temp,
            "watchdog_resets": 0,
            "system_resets": 0,
            "heap_churn_bytes": churn,
            "heap_alloc_failures": 0,
            "distance_cm": round(self.dist, 1),
            "ultrasonic_timeouts": c["timeouts"],
        }
        self.i += 1
        return reading


# ============================================================================
# FALLBACK HEALTH + RSUL ENGINE (used only when the pipeline is not found)
# ============================================================================

LATENCY_THRESHOLD_MS = 100.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_num(v):
    """Safely coerce a pipeline value to float; '' / None / junk -> None."""
    if v is None or isinstance(v, bool) or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class _Window:
    """Rolling window with mean / std / least-squares slope per sample."""

    def __init__(self, maxlen=60):
        self.maxlen = maxlen
        self.w = {}

    def push(self, key, v):
        if v is None:
            return
        self.w.setdefault(key, deque(maxlen=self.maxlen)).append(float(v))

    def arr(self, key):
        return list(self.w.get(key, ()))

    def mean(self, key):
        a = self.arr(key)
        return sum(a) / len(a) if a else 0.0

    def std(self, key):
        a = self.arr(key)
        if len(a) < 2:
            return 0.0
        m = sum(a) / len(a)
        return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))

    def mn(self, key):
        a = self.arr(key)
        return min(a) if a else 0.0

    def mx(self, key):
        a = self.arr(key)
        return max(a) if a else 0.0

    def slope(self, key):
        a = self.arr(key)
        n = len(a)
        if n < 3:
            return 0.0
        sx = sum(range(n))
        sy = sum(a)
        sxy = sum(i * v for i, v in enumerate(a))
        sxx = sum(i * i for i in range(n))
        d = n * sxx - sx * sx
        return 0.0 if d == 0 else (n * sxy - sx * sy) / d

    def rate_per_min(self, key, step):
        return self.slope(key) * (60.0 / step)

    def trend(self, key, eps, step):
        r = self.rate_per_min(key, step)
        if abs(r) < eps:
            return "STABLE"
        return "DEGRADING" if r > 0 else "IMPROVING"


class FallbackHealthEngine:
    """Scores every dimension 0-100 and derives degradation + fault flags."""

    FAULT_NAME = {
        "cpu_overload_flag": "CPU_OVERLOAD",
        "memory_leak_flag": "MEMORY_LEAK",
        "heap_exhaustion_flag": "HEAP_EXHAUSTION",
        "stack_risk_flag": "STACK_RISK",
        "deadline_miss_flag": "DEADLINE_MISS",
        "task_starvation_flag": "TASK_STARVATION",
        "ai_latency_flag": "AI_LATENCY",
        "queue_overflow_flag": "QUEUE_OVERFLOW",
    }

    def __init__(self, step=1.0):
        self.step = step
        self.tt = _Window(60)
        self.deg_ema = None
        self.prev_misses = 0
        self.prev_dropped = 0
        self.fault_count = 0

    def evaluate(self, r):
        tt, step = self.tt, self.step
        tt.push("cpu", r["cpu_utilization"])
        tt.push("heap", r["heap_utilization"])
        tt.push("mem_used", r["used_heap"] / 1024.0)
        tt.push("stack", r["stack_utilization"])
        tt.push("lat", r["inference_time"])
        tt.push("sched", r["scheduler_delay"])
        tt.push("queue", r["queue_utilization"])
        tt.push("misses", r["deadline_misses"])

        uptime_min = max(1 / 60, r["uptime_ms"] / 60000.0)
        miss_rate = r["deadline_misses"] / uptime_min
        duty = r["task_execution_time"] / max(1.0, r["task_period"]) * 100.0

        cpu_rate = tt.rate_per_min("cpu", step)
        heap_rate = tt.rate_per_min("heap", step)
        mem_rate = tt.rate_per_min("mem_used", step)
        stack_rate = tt.rate_per_min("stack", step)
        lat_rate = tt.rate_per_min("lat", step)
        queue_rate = tt.rate_per_min("queue", step)
        sched_rate = tt.rate_per_min("sched", step)

        cpu_s = _clamp(
            100 - max(0, r["cpu_utilization"] - 55) * 1.1 - max(0, cpu_rate) * 2.0,
            0,
            100,
        )
        mem_s = _clamp(
            100
            - max(0, r["heap_utilization"] - 45) * 1.2
            - max(0, heap_rate) * 6.0
            - (25 if r["heap_alloc_failures"] else 0),
            0,
            100,
        )
        stack_s = _clamp(
            100 - max(0, r["stack_utilization"] - 40) * 1.5 - max(0, stack_rate) * 8.0,
            0,
            100,
        )
        task_s = _clamp(
            100
            - max(0, duty - 45) * 1.0
            - max(0, r["scheduler_delay"] - 5) * 1.5
            - max(0, sched_rate) * 2.0,
            0,
            100,
        )
        timing_s = _clamp(
            100
            - max(0, r["task_jitter"] - 1.5) * 8.0
            - max(0, r["scheduler_delay"] - 4) * 2.5
            - miss_rate * 12.0,
            0,
            100,
        )
        ai_s = _clamp(
            100
            - max(0, r["inference_time"] - 35) * 1.4
            - max(0, r["inference_time"] / LATENCY_THRESHOLD_MS - 0.5) * 30
            - max(0, lat_rate) * 3.0,
            0,
            100,
        )
        res_s = _clamp(
            100
            - max(0, r["queue_utilization"] - 45) * 1.4
            - max(0, queue_rate) * 4.0
            - r["dropped_messages"] * 2.0
            - r["ultrasonic_timeouts"] * 1.0,
            0,
            100,
        )

        overall = _clamp(
            cpu_s * 0.20
            + mem_s * 0.18
            + stack_s * 0.10
            + task_s * 0.12
            + timing_s * 0.15
            + ai_s * 0.15
            + res_s * 0.10,
            0,
            100,
        )
        tt.push("overall", overall)

        deg_raw = (100 - overall) / 100.0
        self.deg_ema = (
            deg_raw if self.deg_ema is None else 0.85 * self.deg_ema + 0.15 * deg_raw
        )
        tt.push("deg", self.deg_ema)

        state = (
            "HEALTHY"
            if overall >= 85
            else (
                "WARNING"
                if overall >= 70
                else (
                    "DEGRADED"
                    if overall >= 55
                    else "CRITICAL" if overall >= 40 else "FAILURE_RISK"
                )
            )
        )
        condition = (
            "NORMAL"
            if overall >= 85
            else (
                "MODERATE STRESS"
                if overall >= 70
                else "DEGRADING" if overall >= 55 else "HIGH RISK"
            )
        )

        window_full = len(tt.arr("heap")) >= tt.maxlen
        new_miss = r["deadline_misses"] > self.prev_misses
        new_drop = r["dropped_messages"] > self.prev_dropped
        self.prev_misses = r["deadline_misses"]
        self.prev_dropped = r["dropped_messages"]

        flags = {
            "cpu_overload_flag": r["cpu_utilization"] > 85 or tt.mean("cpu") > 80,
            "memory_leak_flag": window_full and mem_rate > 4.0 and heap_rate > 0.8,
            "heap_exhaustion_flag": r["heap_utilization"] > 90
            or r["free_heap"] < 16384,
            "stack_risk_flag": r["stack_utilization"] > 85
            or (r["stack_utilization"] > 75 and stack_rate > 0.3),
            "deadline_miss_flag": new_miss or miss_rate > 2.0,
            "task_starvation_flag": r["scheduler_delay"] > 40 or duty > 90,
            "ai_latency_flag": r["inference_time"] > 0.8 * LATENCY_THRESHOLD_MS
            or (window_full and lat_rate > 8.0),
            "queue_overflow_flag": r["queue_utilization"] > 90 or new_drop,
        }
        active = [k for k, v in flags.items() if v]
        if active:
            self.fault_count += 1

        out = {
            "cpu_health_score": round(cpu_s, 2),
            "memory_health_score": round(mem_s, 2),
            "stack_health_score": round(stack_s, 2),
            "task_health_score": round(task_s, 2),
            "timing_health_score": round(timing_s, 2),
            "ai_health_score": round(ai_s, 2),
            "resource_health_score": round(res_s, 2),
            "overall_health_score": round(overall, 2),
            "health_state": state,
            "condition_label": condition,
            "degradation_index": round(self.deg_ema, 4),
            "degradation_rate": round(tt.rate_per_min("deg", step), 5),
            "health_change_rate": round(tt.rate_per_min("overall", step), 5),
            "cpu_mean": round(tt.mean("cpu"), 2),
            "cpu_std": round(tt.std("cpu"), 3),
            "cpu_min": round(tt.mn("cpu"), 2),
            "cpu_max": round(tt.mx("cpu"), 2),
            "cpu_trend": tt.trend("cpu", 0.15, step),
            "cpu_growth_rate": round(cpu_rate, 4),
            "memory_mean": round(tt.mean("heap"), 2),
            "memory_std": round(tt.std("heap"), 3),
            "memory_trend": tt.trend("heap", 0.05, step),
            "memory_growth_rate": round(mem_rate, 4),
            "heap_trend": tt.trend("heap", 0.05, step),
            "heap_growth_rate": round(heap_rate, 4),
            "stack_trend": tt.trend("stack", 0.05, step),
            "stack_growth_rate": round(stack_rate, 4),
            "latency_mean": round(tt.mean("lat"), 2),
            "latency_max": round(tt.mx("lat"), 2),
            "latency_trend": tt.trend("lat", 0.3, step),
            "latency_growth_rate": round(lat_rate, 4),
            "task_delay_trend": tt.trend("sched", 0.2, step),
            "deadline_miss_rate": round(miss_rate, 4),
            "deadline_trend": round(tt.rate_per_min("misses", step), 4),
            "queue_trend": tt.trend("queue", 0.2, step),
            "queue_growth_rate": round(queue_rate, 4),
            "duty_cycle_pct": round(duty, 2),
            "fault_type": self.FAULT_NAME[active[0]] if active else "NONE",
            "fault_severity": (
                (
                    "HIGH"
                    if len(active) >= 3
                    else "MEDIUM" if len(active) == 2 else "LOW"
                )
                if active
                else "NONE"
            ),
            "fault_count": self.fault_count,
        }
        out.update(flags)
        return out


class FallbackRsul:
    """Linear-regression RSUL over the run so far, damped by observation length."""

    def __init__(self, model="linear_regression", horizon_min=30, critical=55.0):
        self.model = model
        self.horizon = horizon_min
        self.critical = critical
        self.hist = deque(maxlen=900)  # (uptime_min, overall_health)

    def _slope(self):
        a = list(self.hist)
        if len(a) < 12:
            return 0.0
        n = len(a)
        sx = sum(m for m, _ in a)
        sy = sum(h for _, h in a)
        sxy = sum(m * h for m, h in a)
        sxx = sum(m * m for m, _ in a)
        d = n * sxx - sx * sx
        return 0.0 if d == 0 else (n * sxy - sx * sy) / d

    def predict(self, h, r):
        cur = float(h["overall_health_score"])
        self.hist.append((r["uptime_ms"] / 60000.0, cur))

        observed_min = self.hist[-1][0] - self.hist[0][0] if self.hist else 0.0
        trust = _clamp(observed_min / 60.0, 0.05, 1.0)
        decline = max(0.0, -self._slope()) * trust  # health points lost per minute
        predicted = _clamp(cur - decline * self.horizon, 0, 100)

        fail_p = _clamp(100 / (1 + math.exp((predicted - 52) / 6.0)), 0, 100)

        headroom = max(0.0, cur - self.critical)
        per_hour = decline * 60.0
        rsul = _clamp(headroom / per_hour if per_hour > 0.01 else 720.0, 0, 720)

        fill = min(1.0, len(self.hist) / 60.0)
        vol = _clamp(float(h.get("cpu_std", 0)) / 10.0, 0, 1)
        conf = _clamp((0.55 + 0.4 * fill) * (1 - 0.35 * vol) * 100, 0, 99)

        now = time.time()
        iso = lambda s: time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(s)
        )  # noqa: E731

        dims = [
            ("CPU", h["cpu_health_score"]),
            ("Memory", h["memory_health_score"]),
            ("Stack", h["stack_health_score"]),
            ("Task", h["task_health_score"]),
            ("Timing", h["timing_health_score"]),
            ("AI", h["ai_health_score"]),
            ("Queue", h["resource_health_score"]),
        ]
        deficits = sorted(
            ((k, max(0.0, 100 - float(v))) for k, v in dims), key=lambda x: -x[1]
        )
        total_def = sum(d for _, d in deficits) or 1.0
        dominant = deficits[0][0] if deficits[0][1] > 0.5 else "None"
        contribution = round(deficits[0][1] / total_def * 100, 2)

        risk = (
            "CRITICAL"
            if fail_p >= 60
            else "HIGH" if fail_p >= 30 else "MEDIUM" if fail_p >= 10 else "LOW"
        )
        if risk == "CRITICAL":
            prio = "CRITICAL"
            reco = (
                f"Immediate intervention required — the {dominant.lower()} subsystem is driving "
                f"{contribution:.0f}% of total degradation. Shed load or schedule a controlled restart now."
            )
        elif risk == "HIGH":
            prio = "HIGH"
            reco = (
                f"Schedule maintenance within {max(1, round(rsul / 2))} h. {dominant} is the dominant "
                f"degradation factor ({contribution:.0f}% of deficit) — reduce its duty cycle or re-tune thresholds."
            )
        elif risk == "MEDIUM":
            prio = "MEDIUM"
            reco = (
                f"Monitor {dominant.lower()} closely. Degradation is measurable but slow; predicted health "
                f"in {self.horizon} min is {predicted:.0f}%. No action needed this shift."
            )
        else:
            prio = "LOW"
            reco = (
                f"System operating within nominal envelope. {dominant} contributes most of the (small) "
                f"deficit at {contribution:.0f}%. Continue routine monitoring."
            )

        return {
            "current_health": round(cur, 2),
            "predicted_health": round(predicted, 2),
            "health_state": h["health_state"],
            "current_degradation_rate": h["degradation_rate"],
            "predicted_degradation_rate": round(float(h["degradation_rate"]) * 1.15, 5),
            "failure_probability": round(fail_p, 3),
            "risk_level": risk,
            "predicted_rsul_hours": round(rsul, 2),
            "rsul_confidence": round(conf, 1),
            "predicted_critical_time": iso(now + rsul * 3600),
            "predicted_failure_time": iso(now + rsul * 1.4 * 3600),
            "dominant_degradation_factor": dominant,
            "factor_contribution": contribution,
            "recommendation": reco,
            "recommendation_priority": prio,
            "prediction_horizon_min": self.horizon,
            "observed_window_min": round(observed_min, 2),
            "trend_trust_weight": round(trust, 3),
            "health_decline_per_min": round(decline, 5),
            "compatibility_mode": self.model.upper(),
        }


# ============================================================================
# MINIMAL WEBSOCKET SERVER (stdlib only)
# ============================================================================

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """Build an unmasked server->client frame."""
    head = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head.append(n)
    elif n < 65536:
        head.append(126)
        head += struct.pack(">H", n)
    else:
        head.append(127)
        head += struct.pack(">Q", n)
    return bytes(head) + payload


class Hub:
    """Tracks connected dashboards and fans out JSON messages to them."""

    def __init__(self):
        self.clients = []  # list of (socket, lock)
        self.lock = threading.Lock()
        self.last = None

    def add(self, sock):
        entry = (sock, threading.Lock())
        with self.lock:
            self.clients.append(entry)
        if self.last is not None:
            self.send_one(entry, self.last)
        return entry

    def remove(self, entry):
        with self.lock:
            if entry in self.clients:
                self.clients.remove(entry)

    def count(self):
        with self.lock:
            return len(self.clients)

    def send_one(self, entry, text):
        sock, lk = entry
        frame = ws_frame(text.encode("utf-8"))
        try:
            with lk:
                sock.sendall(frame)
            return True
        except OSError:
            return False

    def broadcast(self, obj, remember=True):
        text = json.dumps(obj)
        if remember and obj.get("type") == "sample":
            self.last = text
        with self.lock:
            targets = list(self.clients)
        for entry in targets:
            if not self.send_one(entry, text):
                self.remove(entry)


HUB = Hub()
HTML_BYTES = b""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PulseOS/1.0"

    def log_message(self, *_args):
        pass  # keep the console clean; the sim loop owns stdout

    # ---- WebSocket upgrade --------------------------------------------------
    def _upgrade(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "missing Sec-WebSocket-Key")
            return
        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode()).digest()
        ).decode()
        self.wfile.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        self.wfile.flush()

        entry = HUB.add(self.connection)
        HUB.send_one(
            entry,
            json.dumps(
                {
                    "type": "status",
                    "meta": {
                        "source": "python",
                        "status": "connected",
                        "detail": "demo_dashboard.py",
                    },
                }
            ),
        )
        try:
            # Drain client frames (pings / close) until the socket drops.
            self.connection.settimeout(None)
            while True:
                data = self.connection.recv(1024)
                if not data:
                    break
                if data[0] & 0x0F == 0x8:  # close opcode
                    break
        except OSError:
            pass
        finally:
            HUB.remove(entry)
            self.close_connection = True

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ws" and "websocket" in (self.headers.get("Upgrade") or "").lower():
            self._upgrade()
            return
        if path in ("/", "/index.html", "/dashboard.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML_BYTES)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(HTML_BYTES)
            return
        self.send_error(404)


# ============================================================================
# MAIN
# ============================================================================


def find_dashboard_html(explicit=None):
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_file():
            sys.exit(f"Dashboard HTML not found: {p}")
        return p
    for p in sorted(HERE.glob("*.html")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "PULSEOS" in text.upper() or "/ws" in text:
            return p
    sys.exit(
        "Could not find the dashboard HTML next to this script. "
        "Pass --html path/to/dashboard.html"
    )


def main():
    global HTML_BYTES

    ap = argparse.ArgumentParser(
        description="Serve the PulseOS dashboard and stream moderate ESP32-S3 telemetry to it"
    )
    ap.add_argument(
        "--minutes", type=float, default=5.0, help="Simulated duration. Default 5"
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wall-clock seconds per sample. 1.0 = real time, 0.25 = 4x. Default 1",
    )
    ap.add_argument(
        "--step", type=float, default=1.0, help="Simulated seconds between samples"
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--html", default=None, help="Path to the dashboard HTML")
    ap.add_argument("--pipeline", default=None, help="Path to your pipeline script")
    ap.add_argument(
        "--require-pipeline",
        action="store_true",
        help="Fail if the pipeline script cannot be found (instead of using the fallback engine)",
    )
    ap.add_argument(
        "--rsul-model",
        choices=["linear_regression", "random_forest"],
        default="linear_regression",
    )
    ap.add_argument("--rsul-threshold", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--loop", action="store_true", help="Restart the run when it finishes"
    )
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    html_path = find_dashboard_html(args.html)
    HTML_BYTES = html_path.read_bytes()

    pipeline = load_pipeline(args.pipeline, required=args.require_pipeline)
    if pipeline is not None:
        pipeline.RAW_CSV_PATH = "demo_raw.csv"
        pipeline.OUTPUT_CSV_PATH = "demo_health.csv"
        pipeline.RSUL_OUTPUT_CSV_PATH = "demo_rsul.csv"
        for name in (
            pipeline.RAW_CSV_PATH,
            pipeline.OUTPUT_CSV_PATH,
            pipeline.RSUL_OUTPUT_CSV_PATH,
        ):
            try:
                os.remove(name)
            except FileNotFoundError:
                pass
        engine = pipeline.HealthIntelligenceEngine()
        tracker = pipeline.TrendTracker()
        source_detail = (
            f"pipeline: {Path(pipeline.__file__).name} · model {args.rsul_model}"
        )
    else:
        engine = tracker = None
        source_detail = "built-in fallback engine (no pipeline script found)"

    total_s = max(10.0, args.minutes * 60.0)
    n_total = max(10, int(total_s / args.step))

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = f"http://{args.host}:{args.port}/"
    print(f"PulseOS dashboard  →  {url}")
    print(f"  serving : {html_path.name}")
    print(f"  source  : {source_detail}")
    print(
        f"  run     : {args.minutes:g} min simulated, {args.speed:g}s per sample"
        + (", looping" if args.loop else "")
    )
    print("  Ctrl-C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    tally = {"overall": [], "rsul": [], "states": Counter()}
    n_done = 0

    try:
        while True:
            gen = ModerateEsp32S3(total_s, step=args.step, seed=args.seed)
            fb_engine = FallbackHealthEngine(args.step) if pipeline is None else None
            fb_rsul = (
                FallbackRsul(args.rsul_model, 30, 55.0) if pipeline is None else None
            )

            for i in range(n_total):
                t0 = time.monotonic()
                reading = gen.next_reading()

                if pipeline is not None:
                    health, rsul, _derived = pipeline.process_reading(
                        engine,
                        tracker,
                        reading,
                        convert_timestamp_ms=False,
                        rsul_model=args.rsul_model,
                        rsul_threshold=args.rsul_threshold,
                    )
                else:
                    health = fb_engine.evaluate(reading)
                    rsul = fb_rsul.predict(health, reading)

                n_done += 1
                overall = to_num(health.get("overall_health_score"))
                if overall is not None:
                    tally["overall"].append(overall)
                rsul_h = to_num(rsul.get("predicted_rsul_hours"))
                if rsul_h is not None:
                    tally["rsul"].append(rsul_h)
                if health.get("health_state"):
                    tally["states"][str(health["health_state"])] += 1

                HUB.broadcast(
                    {
                        "type": "sample",
                        "part1": reading,
                        "part2": health,
                        "part3": rsul,
                        "meta": {
                            "source": "python",
                            "detail": f"{source_detail} · sample {i + 1}/{n_total}",
                            "scenario_label": "Moderate load (scenario 1)",
                            "samples_processed": n_done,
                        },
                    }
                )

                print(
                    f"\r  sample {i + 1:>5}/{n_total}  health {(overall or 0):6.2f}  "
                    f"{str(health.get('health_state', '')):<13} "
                    f"rsul {(rsul_h or 0):7.2f} h  "
                    f"risk {str(rsul.get('risk_level', '')):<9} "
                    f"clients {HUB.count()}   ",
                    end="",
                    flush=True,
                )

                wait = args.speed - (time.monotonic() - t0)
                if wait > 0:
                    time.sleep(wait)

            if not args.loop:
                break
            print("\n  run complete — restarting\n")
    except KeyboardInterrupt:
        pass

    print("\n\nStopped.")
    if tally["overall"]:
        o = tally["overall"]
        print(f"  Samples          : {n_done}")
        print(
            f"  Overall health   : min {min(o):.2f} | avg {sum(o) / len(o):.2f} | max {max(o):.2f}"
        )
    if tally["rsul"]:
        print(f"  Last RSUL        : {tally['rsul'][-1]:.2f} h")
    if tally["states"]:
        print(
            "  States seen      : "
            + ", ".join(f"{k} x{v}" for k, v in tally["states"].most_common())
        )
    if pipeline is not None:
        print(
            f"  Files written    : {pipeline.RAW_CSV_PATH}, {pipeline.OUTPUT_CSV_PATH}, "
            f"{pipeline.RSUL_OUTPUT_CSV_PATH}"
        )
    httpd.shutdown()


if __name__ == "__main__":
    main()
