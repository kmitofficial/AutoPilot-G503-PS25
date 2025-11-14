**AutoPilot VLM — Rover**

**Overview**
- **Project:** Lightweight autonomous rover pipeline combining local perception (camera/optional LiDAR/GPS) with a Vision–Language Model (Google Gemini) for high-level driving guidance.
- **Goal:** Provide a modular, MQTT-based system that runs on a real rover (Jetson Nano) or in simulation on a PC.

**Repository Structure**
- **`cords.txt`**: Waypoint list (latitude longitude altitude, one row per waypoint).
- **`rover.py`**: Camera capture, local perception, MQTT publisher (frames/obstacles/status) and motor command executor.
- **`gemini.py`**: VLM autopilot client — subscribes to `video/stream`, queries Gemini, and publishes `rover/cmd`.
- **`planner.py`**: Builds a navigation graph from `cords.txt` and publishes `rover/waypoints`.
- **`web_ui.py`**: Flask dashboard showing live MJPEG camera feed and telemetry.
- **`requirements.txt`**: Python dependencies for the project.

**Features**
- **Camera Streaming:** Rover publishes frames to `video/stream` (MQTT) for monitoring and VLM input.
- **VLM Autopilot:** Gemini consumes frames + context and returns speed/steering/duration commands.
- **Waypoint Planner:** Uses coordinates in `cords.txt` to build a graph and generate waypoint lists for the rover.
- **MQTT Communication:** Lightweight pub/sub for `video/stream`, `rover/cmd`, `rover/status`, `rover/obstacle`, `rover/waypoints`, `rover/telemetry`.
- **Web Dashboard:** Live video + telemetry via `web_ui.py` (Flask + MJPEG/WS combos).
- **Modular:** Each component (rover, gemini, planner, web UI) runs independently and communicates over MQTT.

**Quick Start — Simulation (Windows / PC)**
1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the Web UI (live feed + telemetry):

```powershell
python web_ui.py
# Visit: http://localhost:5000
```

3. Start the Rover (simulated camera or local webcam):

```powershell
python rover.py
```

4. Start the Gemini VLM autopilot (runs on a PC using your Gemini API key):

```powershell
python gemini.py
```

**Running on Jetson Nano (Real Rover Mode)**
- On the rover (Jetson):
```bash
python3 planner.py
python3 rover.py
```
- On your PC (VLM + dashboard):
```bash
python gemini.py
python web_ui.py
```

**Waypoint Format (`cords.txt`)**
- Each line: `latitude longitude altitude` (space-separated). Example:

```text
17.3970873 78.4897846 500.5
```

**MQTT Topics**
- **`video/stream`**: Base64-encoded JPEG frames from rover camera.
- **`rover/cmd`**: Motor command JSON published by VLM (fields: `speed_left`, `speed_right`, `distance_m`, `duration_s`).
- **`rover/status`**: Rover status events and debugging info.
- **`rover/obstacle`**: Obstacle reports (on-path / any / estimated distance).
- **`rover/waypoints`**: Planner publishes waypoints and path metadata.
- **`rover/telemetry`**: Live telemetry (lat, lon, heading, speed).

**Configuration & API Key**
- **`config.json`**: Optional config file used by scripts for `mqtt` broker, ports, device paths, and tuning constants.
- **Gemini API Key:** `gemini.py` needs a valid Google Gemini API key. For safety, put the key in `config.json` or set an environment variable instead of hard-coding. Example (PowerShell):

```powershell
$env:GENAI_API_KEY = "YOUR_KEY_HERE"
```

**Recommended Runtime Order**
- Ensure your MQTT broker is reachable by all nodes.
- Start `rover.py` on the rover host first (it publishes camera frames and subscribes to commands).
- Start `planner.py` to publish waypoints if using structured navigation.
- Start `gemini.py` (VLM) to consume frames and publish commands.
- Start `web_ui.py` to monitor live feed and telemetry.

**Troubleshooting & Tips**
- If you see many `429` quota errors from Gemini, reduce the VLM call rate (`MIN_INTERVAL` in `gemini.py`) or upgrade your quota.
- On Windows, ensure the camera device index or `cv2.VideoCapture()` source is correct; or set `SHOW_LOCAL=1` to preview frames.
- Tune `DIST_EST_K`, `PATH_WIDTH_FRAC`, and `PATH_BOTTOM_THRESH` in `config.json` for accurate on-path obstacle detection.
- Use simulation mode (set `allow_simulation` in `config.json`) if serial motor controllers or LiDAR are not available.

**Future Improvements**
- YOLO + LiDAR sensor fusion for robust obstacle avoidance.
- Onboard VLM (Qwen-VL, Phi-3-Vision) to reduce latency and quotas.
- Improved planner (A*, RRT*, hybrid) and better sampler-based local planning.
- ROS2 integration and secure MQTT with authentication.

**License & Safety**
- This repository contains example code intended for research and testing. When running on a real vehicle, ensure you have a safety operator and physical kill-switch.

**Contact / Next Steps**
- If you want, I can: wire `get_graph.py` into the planner, add web UI controls (start/stop/auton/manual), or switch Gemini key handling to environment variables.

---

_Quick copy block (paste anywhere):_

```text
git clone <repo>
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python web_ui.py
```
