/*
  AMR Predictive Reliability Monitor - ESP32 Firmware
  ----------------------------------------------------
  Hardware: ESP32 (any dev board with enough RAM, e.g. ESP32-WROOM32)
            + Camera module (e.g. OV2640 / ESP32-CAM) for "Edge AI" perception task
            + HC-SR04 ultrasonic sensor for obstacle-distance ground truth

  What it does:
  - Task A (AI_Task): grabs a camera frame and runs a lightweight "inference"
    (a deliberately simple pixel-processing routine standing in for TFLite Micro).
    Measures inference latency in ms.
  - Task B (MotorControl_Task): simulates the robot's control loop with a strict
    period (deadline). Checks whether AI results arrived in time -> deadline miss count.
  - Task C (Monitor_Task): reads FreeRTOS runtime stats, heap, stack high-water marks,
    ultrasonic distance, computes jitter/queue stats, and prints ONE CSV row per cycle
    over Serial at 115200 baud.
  - A Fault-Injection switch (serial command 'F') artificially slows the AI task down
    over time to simulate degradation, exactly like the project's fault-injection idea.

  CSV COLUMNS (one row per monitoring cycle):
  timestamp_ms,cpu_util_pct,ai_task_cpu_pct,motor_task_cpu_pct,free_heap_kb,
  min_free_heap_kb,ai_task_stack_hwm_bytes,motor_task_stack_hwm_bytes,
  ai_task_exec_ms,motor_task_exec_ms,deadline_misses_total,ai_jitter_ms,
  context_switches_per_s,ai_inference_ms,queue_utilization_pct,dropped_messages_total,
  uptime_s,ultrasonic_distance_cm,health_state

  WHERE TO RUN THIS FILE:
  - Open Arduino IDE (or PlatformIO).
  - Install "ESP32" board package (Boards Manager) and select your ESP32 board.
  - If using ESP32-CAM, install/enable the "esp32cam" / "esp_camera" library that ships
    with the ESP32 core (esp_camera.h). Wire the ultrasonic sensor:
        HC-SR04 TRIG -> GPIO 13
        HC-SR04 ECHO -> GPIO 12 (use a voltage divider, ECHO is 5V logic!)
  - Flash this .ino to the ESP32 over USB.
  - Open Serial Monitor (or just leave it closed) at 115200 baud -- the laptop Python
    script below will read the same USB serial port directly.
*/

#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ---------- Optional camera support ----------
// If you are NOT using an ESP32-CAM board, comment this out and the code will
// fall back to synthetic frame data (still produces valid metrics).
// #define USE_CAMERA
#ifdef USE_CAMERA
#include "esp_camera.h"
#endif

// ---------- Pin config ----------
#define TRIG_PIN 13
#define ECHO_PIN 12

// ---------- Globals shared between tasks ----------
volatile uint32_t g_aiInferenceMs = 50;      // last AI inference latency
volatile uint32_t g_aiTaskExecMs = 0;
volatile uint32_t g_motorTaskExecMs = 0;
volatile uint32_t g_deadlineMisses = 0;
volatile uint32_t g_droppedMessages = 0;
volatile bool     g_aiResultReady = false;
volatile uint32_t g_lastAiJitter = 0;
volatile bool     g_faultInjection = false;   // toggled by serial command 'F'/'N'
volatile uint32_t g_faultLevel = 0;           // increases over time once enabled

// ---------- Queue message structure ----------
struct ObstacleMessage {
  uint32_t latency;
  float distance;
};

QueueHandle_t obstacleQueue;
#define QUEUE_LEN 10

TaskHandle_t aiTaskHandle = NULL;
TaskHandle_t motorTaskHandle = NULL;
TaskHandle_t monitorTaskHandle = NULL;

const TickType_t MOTOR_PERIOD_MS = 100;       // strict control-loop deadline
const TickType_t MONITOR_PERIOD_MS = 1000;    // one CSV row per second

uint32_t contextSwitchCounterLast = 0;

// ---------- Ultrasonic read ----------
float readUltrasonicCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL); // 30ms timeout ~5m range
  if (duration == 0) return -1.0; // no echo / out of range
  return duration * 0.0343f / 2.0f;
}

// ---------- "Edge AI" inference stand-in ----------
// Runs a small deterministic workload whose duration we can inflate via g_faultLevel
// to emulate real inference-time degradation (thermal throttling, memory pressure, etc).
uint32_t runInferenceOnce() {
  uint32_t start = millis();

#ifdef USE_CAMERA
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) {
    // Fake "processing": touch a subset of pixels (stand-in for a TFLite Micro model)
    volatile uint32_t acc = 0;
    for (size_t i = 0; i < fb->len; i += 97) acc += fb->buf[i];
    esp_camera_fb_return(fb);
  }
#else
  // Synthetic workload if no camera attached
  volatile uint32_t acc = 0;
  for (int i = 0; i < 20000; i++) acc += i * i;
#endif

  // Base latency + extra busy-loop to represent fault injection degradation
  uint32_t busyExtraMs = g_faultInjection ? g_faultLevel : 0;
  uint32_t busyStart = millis();
  while (millis() - busyStart < busyExtraMs) { /* burn CPU */ }

  return millis() - start;
}

// ---------- Task: AI / Obstacle Detection ----------
void AI_Task(void *pv) {
  uint32_t lastLatency = 0;
  for (;;) {
    uint32_t t0 = millis();
    uint32_t latency = runInferenceOnce();
    g_aiInferenceMs = latency;
    g_aiTaskExecMs = latency;
    g_lastAiJitter = (lastLatency == 0) ? 0 : abs((int32_t)latency - (int32_t)lastLatency);
    lastLatency = latency;

    float dist = readUltrasonicCM();
    // package a simple "message" (latency + distance) onto the queue for the control task
    ObstacleMessage msg = { latency, dist };
    if (xQueueSend(obstacleQueue, &msg, 0) != pdTRUE) {
      g_droppedMessages++; // queue full -> message dropped
    }
    g_aiResultReady = true;

    // Gradually increase fault level over time once fault injection is enabled,
    // mirroring "increasing workload / thermal effects" from the design doc.
    if (g_faultInjection && g_faultLevel < 200) {
      g_faultLevel += 2; // ms added per cycle, grows over time
    }

    vTaskDelay(pdMS_TO_TICKS(20)); // small yield between inference cycles
  }
}

// ---------- Task: Motor Control ----------
void MotorControl_Task(void *pv) {
  TickType_t lastWake = xTaskGetTickCount();
  ObstacleMessage msg;

  for (;;) {
    uint32_t t0 = millis();
    bool gotMsg = xQueueReceive(obstacleQueue, &msg, 0) == pdTRUE;

    // Deadline check: obstacle info must exist and AI latency must fit within period
    if (!gotMsg || msg.latency > MOTOR_PERIOD_MS) {
      g_deadlineMisses++;
    }

    g_motorTaskExecMs = millis() - t0;

    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(MOTOR_PERIOD_MS));
  }
}

// ---------- Helper: health classification ----------
const char* classifyHealth(uint32_t aiMs, uint32_t deadlineMisses) {
  if (aiMs >= 100 || deadlineMisses > 10) return "CRITICAL";
  if (aiMs >= 70  || deadlineMisses > 2)  return "WARNING";
  return "GOOD";
}

// ---------- Task: Monitoring + CSV output ----------
void Monitor_Task(void *pv) {
  bool headerPrinted = false;
  for (;;) {
    if (!headerPrinted) {
      Serial.println("timestamp_ms,cpu_util_pct,ai_task_cpu_pct,motor_task_cpu_pct,"
                      "free_heap_kb,min_free_heap_kb,ai_task_stack_hwm_bytes,"
                      "motor_task_stack_hwm_bytes,ai_task_exec_ms,motor_task_exec_ms,"
                      "deadline_misses_total,ai_jitter_ms,context_switches_per_s,"
                      "ai_inference_ms,queue_utilization_pct,dropped_messages_total,"
                      "uptime_s,ultrasonic_distance_cm,health_state");
      headerPrinted = true;
    }

    // --- Runtime stats from FreeRTOS ---
    UBaseType_t numTasks = uxTaskGetNumberOfTasks();
    TaskStatus_t *statusArray = (TaskStatus_t *)pvPortMalloc(numTasks * sizeof(TaskStatus_t));
    uint32_t totalRunTime = 0;
    float aiCpuPct = 0, motorCpuPct = 0, overallCpuPct = 0;

    if (statusArray != NULL) {
      numTasks = uxTaskGetSystemState(statusArray, numTasks, &totalRunTime);
      if (totalRunTime > 0) {
        for (UBaseType_t i = 0; i < numTasks; i++) {
          float pct = (100.0f * statusArray[i].ulRunTimeCounter) / totalRunTime;
          if (statusArray[i].xHandle == aiTaskHandle) aiCpuPct = pct;
          if (statusArray[i].xHandle == motorTaskHandle) motorCpuPct = pct;
        }
      }
      vPortFree(statusArray);
    }
    // Approximate overall CPU load as inverse of idle time proxy (simple heuristic)
    overallCpuPct = min(100.0f, aiCpuPct + motorCpuPct + 5.0f);

    uint32_t freeHeap = esp_get_free_heap_size() / 1024;
    uint32_t minFreeHeap = esp_get_minimum_free_heap_size() / 1024;

    UBaseType_t aiStackHwm = aiTaskHandle ? uxTaskGetStackHighWaterMark(aiTaskHandle) * 4 : 0;
    UBaseType_t motorStackHwm = motorTaskHandle ? uxTaskGetStackHighWaterMark(motorTaskHandle) * 4 : 0;

    UBaseType_t queueWaiting = uxQueueMessagesWaiting(obstacleQueue);
    float queueUtilPct = 100.0f * queueWaiting / QUEUE_LEN;

    float distanceCm = readUltrasonicCM();
    uint32_t uptimeS = millis() / 1000;

    const char* health = classifyHealth(g_aiInferenceMs, g_deadlineMisses);

    // NOTE: context switch count is not directly exposed by default FreeRTOS config
    // on ESP32; here we approximate via tick count delta as a placeholder metric.
    uint32_t csPerSec = xTaskGetTickCount() - contextSwitchCounterLast;
    contextSwitchCounterLast = xTaskGetTickCount();

    Serial.printf(
      "%lu,%.1f,%.1f,%.1f,%lu,%lu,%u,%u,%lu,%lu,%lu,%lu,%lu,%lu,%.1f,%lu,%lu,%.1f,%s\n",
      millis(), overallCpuPct, aiCpuPct, motorCpuPct,
      freeHeap, minFreeHeap, aiStackHwm, motorStackHwm,
      g_aiTaskExecMs, g_motorTaskExecMs, g_deadlineMisses, g_lastAiJitter,
      csPerSec, g_aiInferenceMs, queueUtilPct, g_droppedMessages,
      uptimeS, distanceCm, health
    );

    vTaskDelay(pdMS_TO_TICKS(MONITOR_PERIOD_MS));
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

#ifdef USE_CAMERA
  // Fill in your ESP32-CAM pin config here (AI-Thinker pinout shown as example)
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;  config.pin_d1 = 18; config.pin_d2 = 19; config.pin_d3 = 21;
  config.pin_d4 = 36; config.pin_d5 = 39; config.pin_d6 = 34; config.pin_d7 = 35;
  config.pin_xclk = 0; config.pin_pclk = 22; config.pin_vsync = 25;
  config.pin_href = 23; config.pin_sscb_sda = 26; config.pin_sscb_scl = 27;
  config.pin_pwdn = 32; config.pin_reset = -1;
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

void loop() {
  // Accept simple serial commands to control fault injection from the laptop side:
  //   'F' + Enter -> start fault injection (simulate degradation)
  //   'N' + Enter -> reset to normal
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'F') { g_faultInjection = true; }
    if (c == 'N') { g_faultInjection = false; g_faultLevel = 0; }
  }
  vTaskDelay(pdMS_TO_TICKS(50));
}
