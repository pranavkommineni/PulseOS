#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include <DHT.h>

// ---------- Optional camera support ----------
// #define USE_CAMERA
#ifdef USE_CAMERA
#include "esp_camera.h"s
#endif

// ---------- Pin config ----------
#define IR_PIN 34 // IR obstacle sensor digital OUT (input-only pin)
#define DHT_PIN 4 // DHT22 DATA line
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

// ---------- Task periods (ms) ----------
const uint32_t AI_TASK_PERIOD_MS = 20;
const TickType_t MOTOR_PERIOD_MS = 100;
const TickType_t MONITOR_PERIOD_MS = 1000;

// ---------- Queue ----------
struct ObstacleMessage
{
  uint32_t latency;
};
QueueHandle_t obstacleQueue;
#define QUEUE_LEN 10

TaskHandle_t aiTaskHandle = NULL;
TaskHandle_t motorTaskHandle = NULL;
TaskHandle_t monitorTaskHandle = NULL;

portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;

// ---------- Sample / scenario ----------
volatile uint32_t g_sampleId = 0;
volatile uint8_t g_scenarioId = 0; // 0 = normal, 1 = fault-injection active

// ---------- AI task ----------
volatile uint32_t g_aiInferenceMs = 0;
volatile uint32_t g_aiTaskExecMs = 0;
volatile uint32_t g_aiExecCount = 0;
volatile uint32_t g_lastAiJitter = 0;

// Inference stats aggregated over each monitoring window, reset every cycle
volatile uint32_t g_infMin = UINT32_MAX;
volatile uint32_t g_infMax = 0;
volatile uint64_t g_infSum = 0;
volatile uint32_t g_infCountWindow = 0;

// ---------- Motor task ----------
volatile uint32_t g_motorTaskExecMs = 0;
volatile uint32_t g_motorExecCount = 0;
volatile uint32_t g_motorJitter = 0;
volatile uint32_t g_deadlineMisses = 0;

// ---------- Monitor task ----------
volatile uint32_t g_monitorExecCount = 0;

// ---------- Queue / messaging ----------
volatile uint32_t g_messagesSent = 0;
volatile uint32_t g_messagesReceived = 0;
volatile uint32_t g_droppedMessages = 0;

// ---------- Task switches (self-instrumented) ----------
volatile uint32_t g_totalTaskSwitches = 0;

// ---------- Interrupts (IR obstacle sensor used as interrupt source) ----------
volatile uint32_t g_interruptCount = 0;
volatile uint32_t g_lastInterruptLatencyUs = 0;
volatile uint32_t g_irEventUs = 0;
volatile bool g_irEventPending = false;

// ---------- Temperature (DHT22) ----------
volatile float g_lastTempC = -999.0f; // sentinel: no valid reading yet
uint32_t g_lastDhtReadMs = 0;
const uint32_t DHT_MIN_INTERVAL_MS = 2200; // DHT22 needs >=2s between reads

// ---------- Fault injection ----------
volatile bool g_faultInjection = false;
volatile uint32_t g_faultLevel = 0;

// ---------- Persistent reset counters (RTC memory: survive reset, not power loss) ----------
RTC_DATA_ATTR uint32_t rtc_systemResets = 0;
RTC_DATA_ATTR uint32_t rtc_watchdogResets = 0;

uint32_t contextSwitchCounterLast = 0;

// ================= IR obstacle sensor (interrupt source) =================
// Fires on every state change (obstacle appears / clears) -- an asynchronous
// event, unlike the old ultrasonic trigger/echo round-trip. interrupt_latency
// is therefore redefined here as real ISR-to-task response latency: the time
// between the interrupt firing and the point where AI_Task notices and
// services it. That's what "interrupt_latency" should mean, and it no longer
// risks a 30ms block if a sensor isn't wired.
void IRAM_ATTR irObstacleISR()
{
  g_irEventUs = micros();
  g_irEventPending = true;
  g_interruptCount++;
}

// ================= "Edge AI" inference stand-in =================
// Deterministic workload, duration inflatable via g_faultLevel to emulate
// real inference-time degradation (thermal throttling, memory pressure, etc).
uint32_t runInferenceOnce()
{
  uint32_t start = millis();

#ifdef USE_CAMERA
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb)
  {
    volatile uint32_t acc = 0;
    for (size_t i = 0; i < fb->len; i += 97)
      acc += fb->buf[i];
    esp_camera_fb_return(fb);
  }
#else
  volatile uint32_t acc = 0;
  for (int i = 0; i < 20000; i++)
    acc += i * i;
#endif

  uint32_t busyExtraMs = g_faultInjection ? g_faultLevel : 0;
  uint32_t busyStart = millis();
  while (millis() - busyStart < busyExtraMs)
  { /* burn CPU */
  }

  return millis() - start;
}

// ================= Tasks =================

void AI_Task(void *pv)
{
  uint32_t lastLatency = 0;
  for (;;)
  {
    g_totalTaskSwitches++;

    uint32_t latency = runInferenceOnce();
    g_aiInferenceMs = latency;
    g_aiTaskExecMs = latency;
    g_aiExecCount++;

    g_lastAiJitter = (lastLatency == 0) ? 0 : abs((int32_t)latency - (int32_t)lastLatency);
    lastLatency = latency;

    portENTER_CRITICAL(&g_mux);
    if (latency < g_infMin)
      g_infMin = latency;
    if (latency > g_infMax)
      g_infMax = latency;
    g_infSum += latency;
    g_infCountWindow++;
    portEXIT_CRITICAL(&g_mux);

    // Service any pending IR-obstacle interrupt: real ISR-to-task latency
    if (g_irEventPending)
    {
      g_lastInterruptLatencyUs = micros() - g_irEventUs;
      g_irEventPending = false;
    }

    ObstacleMessage msg = {latency};
    if (xQueueSend(obstacleQueue, &msg, 0) == pdTRUE)
    {
      g_messagesSent++;
    }
    else
    {
      g_droppedMessages++;
    }

    if (g_faultInjection && g_faultLevel < 200)
    {
      g_faultLevel += 2; // ms added per cycle, grows over time
    }

    vTaskDelay(pdMS_TO_TICKS(AI_TASK_PERIOD_MS));
  }
}

void MotorControl_Task(void *pv)
{
  TickType_t lastWake = xTaskGetTickCount();
  ObstacleMessage msg;
  uint32_t lastExec = 0;

  for (;;)
  {
    g_totalTaskSwitches++;
    uint32_t t0 = millis();

    bool gotMsg = xQueueReceive(obstacleQueue, &msg, 0) == pdTRUE;
    if (gotMsg)
      g_messagesReceived++;

    if (!gotMsg || msg.latency > MOTOR_PERIOD_MS)
    {
      g_deadlineMisses++;
    }

    g_motorTaskExecMs = millis() - t0;
    g_motorExecCount++;
    g_motorJitter = (lastExec == 0) ? 0 : abs((int32_t)g_motorTaskExecMs - (int32_t)lastExec);
    lastExec = g_motorTaskExecMs;

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(MOTOR_PERIOD_MS));
  }
}

const char *taskStateToStr(eTaskState s)
{
  switch (s)
  {
  case eRunning:
    return "RUNNING";
  case eReady:
    return "READY";
  case eBlocked:
    return "BLOCKED";
  case eSuspended:
    return "SUSPENDED";
  case eDeleted:
    return "DELETED";
  default:
    return "UNKNOWN";
  }
}

float estimatePowerConsumption(float cpuUtilPct)
{
  // ESTIMATE ONLY -- no current-sense hardware (e.g. INA219) wired to this
  // board. Rough linear model: idle draw + load-proportional draw.
  const float IDLE_MW = 120.0f;
  const float MAX_EXTRA_MW = 350.0f;
  return IDLE_MW + (cpuUtilPct / 100.0f) * MAX_EXTRA_MW;
}

float readSystemTemperature()
{
  // Real DHT22 reading, rate-limited to the sensor's minimum ~2s interval.
  // Between reads (and on a failed read, which DHT22 does occasionally),
  // reuses the last good value instead of blocking or logging garbage/NaN.
  uint32_t now = millis();
  if (now - g_lastDhtReadMs >= DHT_MIN_INTERVAL_MS)
  {
    float t = dht.readTemperature(); // Celsius
    g_lastDhtReadMs = now;
    if (!isnan(t))
    {
      g_lastTempC = t;
    }
  }
  return g_lastTempC;
}

void Monitor_Task(void *pv)
{
  bool headerPrinted = false;
  TickType_t lastWake = xTaskGetTickCount();
  uint32_t expectedWakeMs = millis();
  static uint32_t lastMonitorExec = 0;

  for (;;)
  {
    g_totalTaskSwitches++;
    uint32_t cycleStartMs = millis();
    uint32_t schedulerDelayMs = (cycleStartMs >= expectedWakeMs) ? (cycleStartMs - expectedWakeMs) : 0;

    if (!headerPrinted)
    {
      Serial.println(
          "timestamp,sample_id,uptime_ms,scenario_id,cpu_utilization,cpu_idle,"
          "task_cpu_utilization,total_heap,free_heap,used_heap,heap_utilization,"
          "minimum_free_heap,stack_high_water_mark,stack_utilization,task_name,"
          "task_priority,task_state,task_execution_time,task_period,task_jitter,"
          "task_execution_count,deadline_misses,context_switches,task_switches,"
          "scheduler_delay,active_task_count,inference_time,min_inference_time,"
          "max_inference_time,average_inference_time,inference_count,"
          "inference_frequency,queue_length,queue_capacity,queue_utilization,"
          "messages_sent,messages_received,dropped_messages,interrupt_count,"
          "interrupt_latency,power_consumption,system_temperature,watchdog_resets,"
          "system_resets");
      headerPrinted = true;
    }

    g_sampleId++;
    g_monitorExecCount++;

    // ---- system-wide stats, computed once per cycle ----
    UBaseType_t numTasks = uxTaskGetNumberOfTasks();
    TaskStatus_t *statusArray = (TaskStatus_t *)pvPortMalloc(numTasks * sizeof(TaskStatus_t));
    uint32_t totalRunTime = 0;
    float aiCpuPct = 0, motorCpuPct = 0, monitorCpuPct = 0, overallCpuPct = 0;

    if (statusArray != NULL)
    {
      numTasks = uxTaskGetSystemState(statusArray, numTasks, &totalRunTime);
      if (totalRunTime > 0)
      {
        for (UBaseType_t i = 0; i < numTasks; i++)
        {
          float pct = (100.0f * statusArray[i].ulRunTimeCounter) / totalRunTime;
          if (statusArray[i].xHandle == aiTaskHandle)
            aiCpuPct = pct;
          else if (statusArray[i].xHandle == motorTaskHandle)
            motorCpuPct = pct;
          else if (statusArray[i].xHandle == monitorTaskHandle)
            monitorCpuPct = pct;
        }
      }
      vPortFree(statusArray);
    }
    overallCpuPct = min(100.0f, aiCpuPct + motorCpuPct + monitorCpuPct + 3.0f); // +3 idle/system fudge
    float cpuIdlePct = 100.0f - overallCpuPct;

    uint32_t freeHeap = esp_get_free_heap_size();
    uint32_t totalHeap = ESP.getHeapSize();
    uint32_t usedHeap = totalHeap - freeHeap;
    float heapUtilPct = 100.0f * usedHeap / totalHeap;
    uint32_t minFreeHeap = esp_get_minimum_free_heap_size();

    uint32_t nowTicks = xTaskGetTickCount();
    uint32_t contextSwitches = nowTicks - contextSwitchCounterLast; // approx: RTOS ticks elapsed
    contextSwitchCounterLast = nowTicks;

    UBaseType_t queueWaiting = uxQueueMessagesWaiting(obstacleQueue);
    float queueUtilPct = 100.0f * queueWaiting / QUEUE_LEN;

    portENTER_CRITICAL(&g_mux);
    uint32_t infMin = (g_infCountWindow > 0) ? g_infMin : 0;
    uint32_t infMax = g_infMax;
    uint32_t infCount = g_infCountWindow;
    float infAvg = (g_infCountWindow > 0) ? ((float)g_infSum / g_infCountWindow) : 0;
    g_infMin = UINT32_MAX;
    g_infMax = 0;
    g_infSum = 0;
    g_infCountWindow = 0;
    portEXIT_CRITICAL(&g_mux);

    float infFrequencyHz = infCount / (MONITOR_PERIOD_MS / 1000.0f);

    float powerMw = estimatePowerConsumption(overallCpuPct);
    float tempC = readSystemTemperature();

    uint32_t uptimeMs = millis();
    unsigned long tsMs = millis();

    uint32_t monitorExecMs = millis() - cycleStartMs; // work time so far this cycle
    uint32_t monitorJitter = (lastMonitorExec == 0) ? 0 : abs((int32_t)monitorExecMs - (int32_t)lastMonitorExec);
    lastMonitorExec = monitorExecMs;

    struct TaskRow
    {
      const char *name;
      float cpuPct;
      uint32_t execMs;
      uint32_t period;
      uint32_t jitter;
      uint32_t execCount;
      uint32_t allocatedStackBytes;
      TaskHandle_t handle;
    };

    TaskRow rows[3] = {
        {"AI_Task", aiCpuPct, g_aiTaskExecMs, AI_TASK_PERIOD_MS, g_lastAiJitter, g_aiExecCount, 8192, aiTaskHandle},
        {"MotorControl_Task", motorCpuPct, g_motorTaskExecMs, MOTOR_PERIOD_MS, g_motorJitter, g_motorExecCount, 4096, motorTaskHandle},
        {"Monitor_Task", monitorCpuPct, monitorExecMs, MONITOR_PERIOD_MS, monitorJitter, g_monitorExecCount, 4096, monitorTaskHandle}};

    for (int i = 0; i < 3; i++)
    {
      TaskHandle_t h = rows[i].handle;
      UBaseType_t hwmWords = h ? uxTaskGetStackHighWaterMark(h) : 0;
      uint32_t hwmBytes = hwmWords * 4;
      float stackUtilPct = 100.0f * (rows[i].allocatedStackBytes - (float)hwmBytes) / rows[i].allocatedStackBytes;
      eTaskState state = h ? eTaskGetState(h) : eInvalid;
      UBaseType_t prio = h ? uxTaskPriorityGet(h) : 0;

      Serial.printf(
          "%lu,%lu,%lu,%u,%.1f,%.1f,%.1f,%lu,%lu,%lu,%.1f,%lu,%lu,%.1f,%s,%u,%s,%lu,%lu,%lu,"
          "%lu,%lu,%lu,%lu,%lu,%u,%lu,%lu,%lu,%.1f,%lu,%.1f,%u,%d,%.1f,%lu,%lu,%lu,%lu,%lu,"
          "%.1f,%.1f,%lu,%lu\n",
          tsMs, (unsigned long)g_sampleId, (unsigned long)uptimeMs, (unsigned)g_scenarioId,
          overallCpuPct, cpuIdlePct,
          rows[i].cpuPct, (unsigned long)totalHeap, (unsigned long)freeHeap, (unsigned long)usedHeap, heapUtilPct,
          (unsigned long)minFreeHeap, (unsigned long)hwmBytes, stackUtilPct, rows[i].name,
          (unsigned)prio, taskStateToStr(state), (unsigned long)rows[i].execMs, (unsigned long)rows[i].period, (unsigned long)rows[i].jitter,
          (unsigned long)rows[i].execCount, (unsigned long)g_deadlineMisses, (unsigned long)contextSwitches, (unsigned long)g_totalTaskSwitches,
          (unsigned long)schedulerDelayMs, (unsigned)numTasks, (unsigned long)g_aiInferenceMs, (unsigned long)infMin,
          (unsigned long)infMax, infAvg, (unsigned long)infCount,
          infFrequencyHz, (unsigned)queueWaiting, (int)QUEUE_LEN, queueUtilPct,
          (unsigned long)g_messagesSent, (unsigned long)g_messagesReceived, (unsigned long)g_droppedMessages, (unsigned long)g_interruptCount,
          (unsigned long)g_lastInterruptLatencyUs, powerMw, tempC, (unsigned long)rtc_watchdogResets,
          (unsigned long)rtc_systemResets);
    }

    expectedWakeMs += MONITOR_PERIOD_MS;
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(MONITOR_PERIOD_MS));
  }
}

// ================= Boot-time reset reason tracking =================

void classifyAndCountResetReason()
{
  esp_reset_reason_t reason = esp_reset_reason();
  rtc_systemResets++; // counts every boot (persists across reset, not power loss)
  switch (reason)
  {
  case ESP_RST_WDT:
  case ESP_RST_INT_WDT:
  case ESP_RST_TASK_WDT:
    rtc_watchdogResets++;
    break;
  default:
    break;
  }
}

void setup()
{
  Serial.begin(115200);
  delay(500);

  classifyAndCountResetReason();

  pinMode(IR_PIN, INPUT); // GPIO34 is input-only, no internal pull available
  attachInterrupt(digitalPinToInterrupt(IR_PIN), irObstacleISR, CHANGE);

  dht.begin();

#ifdef USE_CAMERA
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QQVGA;
  config.fb_count = 1;
  esp_camera_init(&config);
#endif

  obstacleQueue = xQueueCreate(QUEUE_LEN, sizeof(ObstacleMessage));

  xTaskCreatePinnedToCore(AI_Task, "AI_Task", 8192, NULL, 2, &aiTaskHandle, 1);
  xTaskCreatePinnedToCore(MotorControl_Task, "MotorControl_Task", 4096, NULL, 3, &motorTaskHandle, 1);
  xTaskCreatePinnedToCore(Monitor_Task, "Monitor_Task", 4096, NULL, 1, &monitorTaskHandle, 0);
}

void loop()
{
  // Serial commands from the laptop side:
  //   'F' + Enter -> start fault injection (simulate degradation)
  //   'N' + Enter -> reset to normal
  if (Serial.available())
  {
    char c = Serial.read();
    if (c == 'F')
    {
      g_faultInjection = true;
      g_scenarioId = 1;
    }
    if (c == 'N')
    {
      g_faultInjection = false;
      g_faultLevel = 0;
      g_scenarioId = 0;
    }
  }
  vTaskDelay(pdMS_TO_TICKS(50));
}
