### Work Update

* Developed ESP32-based **AMR Predictive Reliability Monitoring** firmware using FreeRTOS.
* Implemented separate **AI, Motor Control, and Monitoring tasks** with inter-task communication using queues.
* Integrated **HC-SR04 ultrasonic sensing** for obstacle-distance monitoring.
* Added monitoring of **AI latency, CPU usage, heap, stack, jitter, deadline misses, queue utilization, and system health**.
* Implemented **fault injection** to simulate gradual AI performance degradation.
* Developed a **Python laptop data logger** using PySerial to collect ESP32 telemetry and generate CSV/Excel datasets.
* Debugged ESP32 queue compilation issues and established **USB serial connection on COM9**.
* Currently debugging **ESP32 telemetry transmission to the Python logger**.
* Developed and committed **Health Intelligence Engine**.