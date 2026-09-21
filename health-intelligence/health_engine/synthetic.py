import math
import random
from typing import List, Dict, Any, Optional


class SyntheticTelemetryGenerator:
    """
    Generates correlated multi-metric telemetry streams for Person 2 verification.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.total_heap = 1048576  # 1 MB Heap
        self.queue_capacity = 100

    def _base_reading(
        self, sample_id: int, timestamp: float, uptime_ms: float, scenario_id: str = "1"
    ) -> Dict[str, Any]:
        """Generate baseline healthy values with slight natural noise."""
        cpu = round(self.rng.uniform(22.0, 32.0), 2)
        idle = round(100.0 - cpu, 2)
        task_cpu = round(cpu * 0.45, 2)

        used_h = int(self.total_heap * self.rng.uniform(0.18, 0.24))
        free_h = self.total_heap - used_h
        heap_u = round((used_h / self.total_heap) * 100.0, 2)
        min_free = int(free_h * 0.92)

        stack_u = round(self.rng.uniform(30.0, 42.0), 2)
        stack_hwm = 4096 - int(stack_u * 30)

        inf_t = round(self.rng.uniform(32.0, 37.0), 2)
        min_inf = round(inf_t - self.rng.uniform(2.0, 4.0), 2)
        max_inf = round(inf_t + self.rng.uniform(3.0, 6.0), 2)
        avg_inf = inf_t

        q_len = int(self.rng.uniform(5, 15))
        q_util = round((q_len / self.queue_capacity) * 100.0, 2)

        return {
            "timestamp": round(timestamp, 3),
            "sample_id": sample_id,
            "uptime_ms": round(uptime_ms, 1),
            "scenario_id": scenario_id,
            "cpu_utilization": cpu,
            "cpu_idle": idle,
            "task_cpu_utilization": task_cpu,
            "total_heap": self.total_heap,
            "free_heap": free_h,
            "used_heap": used_h,
            "heap_utilization": heap_u,
            "minimum_free_heap": min_free,
            "stack_high_water_mark": stack_hwm,
            "stack_utilization": stack_u,
            "task_name": "ai_inference_task",
            "task_priority": 3,
            "task_state": "RUNNING",
            "task_execution_time": 18.5,
            "task_period": 50.0,
            "task_jitter": round(self.rng.uniform(0.5, 2.0), 2),
            "task_execution_count": sample_id * 20,
            "deadline_misses": 0,
            "context_switches": sample_id * 50,
            "task_switches": sample_id * 40,
            "scheduler_delay": round(self.rng.uniform(1.0, 3.5), 2),
            "active_task_count": 6,
            "inference_time": inf_t,
            "min_inference_time": min_inf,
            "max_inference_time": max_inf,
            "average_inference_time": avg_inf,
            "inference_count": sample_id * 5,
            "inference_frequency": 10.0,
            "queue_length": q_len,
            "queue_capacity": self.queue_capacity,
            "queue_utilization": q_util,
            "messages_sent": sample_id * 15,
            "messages_received": sample_id * 15,
            "dropped_messages": 0,
            "interrupt_count": sample_id * 120,
            "interrupt_latency": round(self.rng.uniform(0.2, 0.6), 3),
            "power_consumption": round(self.rng.uniform(1400.0, 1600.0), 1),
            "system_temperature": round(self.rng.uniform(42.0, 47.0), 1),
            "watchdog_resets": 0,
            "system_resets": 0,
        }

    def generate_scenario(
        self,
        scenario_name: str,
        num_samples: int = 20,
        start_time: float = 1000.0,
        step_sec: float = 1.0,
        scenario_id: str = "1",
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete time series of correlated telemetry records for a named scenario.
        """
        records = []
        cum_dropped = 0
        cum_misses = 0

        for i in range(num_samples):
            t = start_time + i * step_sec
            uptime = (t - start_time) * 1000.0 + 5000.0
            rec = self._base_reading(
                sample_id=i + 1, timestamp=t, uptime_ms=uptime, scenario_id=scenario_id
            )
            progress = i / max(1, num_samples - 1)

            if scenario_name == "HEALTHY":
                pass  # Baseline remains healthy

            elif scenario_name == "CPU_OVERLOAD":
                # CPU climbs to 94%, idle collapses, thermal rises, power rises
                cpu = min(98.0, 40.0 + progress * 56.0 + self.rng.uniform(-1.0, 1.0))
                rec["cpu_utilization"] = round(cpu, 2)
                rec["cpu_idle"] = round(100.0 - cpu, 2)
                rec["task_cpu_utilization"] = round(cpu * 0.7, 2)
                rec["scheduler_delay"] = round(3.0 + progress * 16.0, 2)
                rec["system_temperature"] = round(45.0 + progress * 28.0, 1)
                rec["power_consumption"] = round(1500.0 + progress * 1400.0, 1)

            elif scenario_name == "MEMORY_LEAK":
                # Used heap grows steadily from 20% to 88%, free heap drops, min_free drops
                used = int(self.total_heap * (0.22 + progress * 0.64))
                free = self.total_heap - used
                rec["used_heap"] = used
                rec["free_heap"] = free
                rec["heap_utilization"] = round((used / self.total_heap) * 100.0, 2)
                rec["minimum_free_heap"] = free

            elif scenario_name == "HEAP_EXHAUSTION":
                # Rapid climb past 94% with critical minimum free heap
                used = int(self.total_heap * (0.60 + progress * 0.36))
                free = self.total_heap - used
                rec["used_heap"] = used
                rec["free_heap"] = free
                rec["heap_utilization"] = round((used / self.total_heap) * 100.0, 2)
                rec["minimum_free_heap"] = free

            elif scenario_name == "STACK_RISK":
                # Stack utilization climbs to 94%, watermark collapses below critical floor
                stack_u = min(95.0, 40.0 + progress * 54.0)
                rec["stack_utilization"] = round(stack_u, 2)
                rec["stack_high_water_mark"] = max(
                    32, int(4096 * (1.0 - stack_u / 100.0))
                )

            elif scenario_name == "DEADLINE_DEGRADATION":
                # Execution time exceeds period (55ms > 50ms), deadline misses accumulate
                rec["task_execution_time"] = round(30.0 + progress * 28.0, 2)
                if rec["task_execution_time"] > rec["task_period"]:
                    cum_misses += int(1 + progress * 3)
                rec["deadline_misses"] = cum_misses
                rec["scheduler_delay"] = round(4.0 + progress * 22.0, 2)
                rec["task_jitter"] = round(2.0 + progress * 12.0, 2)

            elif scenario_name == "TASK_STARVATION":
                # Target task is READY, but receives 0 CPU while scheduler delay spikes
                rec["task_state"] = "READY"
                rec["task_cpu_utilization"] = 0.0
                rec["cpu_utilization"] = round(70.0 + progress * 8.0, 2)
                rec["cpu_idle"] = round(100.0 - rec["cpu_utilization"], 2)
                rec["scheduler_delay"] = round(18.0 + progress * 28.0, 2)

            elif scenario_name == "AI_LATENCY":
                # Inference latency inflates to 85-110ms (>2x nominal 35ms)
                inf_t = round(35.0 + progress * 55.0 + self.rng.uniform(-2, 3), 2)
                rec["inference_time"] = inf_t
                rec["average_inference_time"] = inf_t
                rec["max_inference_time"] = round(inf_t + 12.0, 2)
                rec["min_inference_time"] = round(inf_t - 8.0, 2)
                rec["cpu_utilization"] = round(30.0 + progress * 35.0, 2)
                rec["cpu_idle"] = round(100.0 - rec["cpu_utilization"], 2)

            elif scenario_name == "QUEUE_CONGESTION":
                # Queue length fills to capacity, dropped messages begin accumulating
                q_len = min(100, int(15 + progress * 88))
                rec["queue_length"] = q_len
                rec["queue_utilization"] = round(
                    (q_len / self.queue_capacity) * 100.0, 2
                )
                if q_len >= 95:
                    cum_dropped += int(1 + progress * 4)
                rec["dropped_messages"] = cum_dropped

            elif scenario_name == "MULTIPLE_FAULTS":
                # Simultaneous CPU overload, AI latency degradation, and queue congestion
                cpu = min(96.0, 75.0 + progress * 22.0)
                rec["cpu_utilization"] = round(cpu, 2)
                rec["cpu_idle"] = round(100.0 - cpu, 2)
                inf_t = round(55.0 + progress * 30.0, 2)
                rec["inference_time"] = inf_t
                rec["average_inference_time"] = inf_t
                q_len = min(100, int(80 + progress * 18))
                rec["queue_length"] = q_len
                rec["queue_utilization"] = round(
                    (q_len / self.queue_capacity) * 100.0, 2
                )
                if q_len >= 95:
                    rec["dropped_messages"] = int(1 + progress * 6)

            elif scenario_name == "RECOVERY":
                # First half degraded, second half recovers to nominal
                if i < num_samples // 2:
                    cpu = 92.0
                    lat = 80.0
                    q = 90
                else:
                    recov_progress = (i - num_samples // 2) / max(1, num_samples // 2)
                    cpu = 92.0 - recov_progress * 62.0
                    lat = 80.0 - recov_progress * 46.0
                    q = int(90 - recov_progress * 75)

                rec["cpu_utilization"] = round(cpu, 2)
                rec["cpu_idle"] = round(100.0 - cpu, 2)
                rec["inference_time"] = round(lat, 2)
                rec["average_inference_time"] = round(lat, 2)
                rec["queue_length"] = q
                rec["queue_utilization"] = round((q / self.queue_capacity) * 100.0, 2)

            records.append(rec)

        return records
