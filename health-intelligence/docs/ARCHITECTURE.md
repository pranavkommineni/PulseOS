# Person 2 System Architecture & Mathematical Foundations

## 1. System Overview & Context

In an RTOS-based Edge-AI system:
- **Person 1**: Collects low-level hardware counters, scheduler hooks, task control blocks, and inference timers.
- **Person 2**: Ingests the raw stream, validates and cleans corrupted readings, maintains rolling historical windows, detects device resets, computes moving statistics and OLS regression trends, calculates 0–100 domain health scores, flags faults, diagnoses root causes, and outputs structured health intelligence.
- **Person 3**: Consumes Person 2's health intelligence for predictive maintenance, task throttling, scenario adaptation, or failover.

```
Person 1 (Raw Telemetry: 44 fields)
        ↓
+-------------------------------------------------------------+
| PERSON 2 HEALTH INTELLIGENCE ENGINE                         |
|                                                             |
|  [InputValidator] -> Physical boundaries & reconciliation   |
|  [HistoryManager] -> Time-based rolling buffer & reset check|
|  [FeatureEngineer]-> Uncorrelated derived signals           |
|  [StatisticsEngine]-> OLS slopes, growth rates, moments     |
|  [HealthScoreEngine]-> 0-100 scores & bottleneck aggregation|
|  [FaultDetector]  -> 8 Multi-metric flags & root cause      |
|  [Degradation]    -> Degradation index, rates & states      |
|  [OutputFormatter]-> Strict 47-key JSON contract            |
|                                                             |
|  [Diagnostics]    -> Non-intrusive mathematical explainers  |
+-------------------------------------------------------------+
        ↓
Person 3 (Health Intelligence: 47 fields)
```

---

## 2. Mathematical Rigor & Algorithms

### 2.1 Trend Estimation via Ordinary Least Squares (OLS)
Rather than instantaneous two-point delta ($y_t - y_{t-1}$), which is vulnerable to high-frequency sensor noise, trends are estimated using the OLS regression slope over timestamp $t$ (in seconds):

$$m = \frac{\sum_{i=1}^N (t_i - \bar{t})(y_i - \bar{y})}{\sum_{i=1}^N (t_i - \bar{t})^2}$$

Trend classification uses calibrated slope boundaries:
- If $N < N_{\text{min}}$ (e.g. 5 points): `INSUFFICIENT_DATA`.
- For metrics where increasing is adverse (CPU, memory, latency, queue, delay, misses):
  - $m > +T_{\text{slope}} \implies \text{"DEGRADING"}$
  - $m < -T_{\text{slope}} \implies \text{"IMPROVING"}$
  - $|m| \le T_{\text{slope}} \implies \text{"STABLE"}$

### 2.2 Growth Rate Calculation & Sign Conventions
Growth rates represent the normalized temporal rate of change (% per second):

$$g = \frac{m}{\|\bar{y}\| + \epsilon} \times 100\%$$

**Universal Sign Convention**:
- **Positive ($g > 0$)**: The physical metric value is growing over time.
- **Negative ($g < 0$)**: The physical metric value is shrinking over time.
- **Zero ($g = 0$)**: The physical metric value is constant.

### 2.3 Bottleneck-Constrained Health Aggregation
Linear weighted averaging can obscure single-subsystem catastrophic failure (e.g., CPU at 100%, memory at 99%, but thermal and queue normal). To prevent masked failures while honoring scenario-configured weights:

$$\bar{H} = \frac{\sum_{k=1}^K w_k H_k}{\sum_{k=1}^K w_k}$$

$$H_{\text{overall}} = \begin{cases} 
\bar{H}, & \text{if } \min_k(H_k) \ge 40.0 \\
\min\left(\bar{H}, H_{\min} + 0.35(\bar{H} - H_{\min})\right), & \text{if } \min_k(H_k) < 40.0
\end{cases}$$

### 2.4 Degradation Index vs Health Change Rate
Person 2 distinguishes between the system degradation index, degradation rate, and health change rate:
1. **Degradation Index ($D \in [0.0, 1.0]$)**:
   $$D = \min\left(1.0, 0.75 \cdot \frac{100 - H_{\text{overall}}}{100} + 0.04 \cdot N_{\text{faults}} + P_{\text{severity}} + P_{\text{trend}}\right)$$
2. **Degradation Rate ($dD/dt$, units: $1/\text{sec}$)**:
   OLS regression slope of $D$ over time. Positive = degradation worsening.
3. **Health Change Rate ($dH/dt$, units: $\text{points}/\text{sec}$)**:
   OLS regression slope of $H_{\text{overall}}$ over time. Positive = health improving.

---

## 3. Calibration Parameters vs Fixed Relationships

### Fixed Physical & Logical Invariants
- `cpu_utilization + cpu_idle = 100.0`
- `used_heap + free_heap = total_heap`
- `heap_utilization = (used_heap / total_heap) * 100.0`
- `queue_utilization = (queue_length / queue_capacity) * 100.0`
- `min_inference_time <= average_inference_time <= max_inference_time`
- Non-negativity of execution times, counters, and memory bytes

### Calibratable Parameters (`health_engineConfig`)
All thresholds, safety margins, time constants, and scenario profiles are centralized in `health_engine/config.py`:
- `window_duration_sec`: Rolling analysis duration (default: 120s)
- `min_trend_points`: Minimum points before trend calculation (default: 5)
- `overload_threshold_pct`: CPU saturation threshold (default: 88%)
- `heap_exhaustion_pct`: Critical memory floor (default: 92%)
- `critical_min_free_heap_bytes`: Minimum free heap reserve (default: 8192 B)
- `stack_risk_pct`: Critical stack utilization threshold (default: 88%)
- `nominal_inference_time_ms`: Baseline AI execution latency (default: 35ms)
- `latency_critical_factor`: AI degradation multiplier (default: 1.80x)
- `ScenarioProfile`: Domain weighting configurations per scenario_id

---

## 4. Explainability & Diagnostics Mode

In compliance with Section 30, Person 2 includes a non-intrusive diagnostic interface:
- Production calls: `process_reading(input_json)` returns strictly the 47 production fields.
- Diagnostic calls: `process_reading(input_json, return_diagnostics=True)` returns `(production_output, diagnostic_traces)`.
- Or call `engine.get_diagnostics(metric_name)`.

Diagnostic traces record:
- Exact mathematical formula applied
- Input metrics and derived features used
- Normalized intermediate values
- Scenario weight contributions
- Number of historical points in window
- Sample integrity confidence score ($0.0 \le C \le 1.0$)
- Natural language explanation of the resulting decision
