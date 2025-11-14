🚗 AutoPilot VLM – Rover

Autonomous rover navigation powered by a Vision–Language Model (Google Gemini) and local perception (camera, GPS, LiDAR).
The system uses MQTT for real-time communication between the rover (Jetson Nano), the VLM inference module (PC), and a live monitoring dashboard (Flask Web UI).

📌 Features

🔭 Camera Streaming — Rover captures frames and publishes via MQTT

🧠 VLM Autopilot — Gemini processes frames → generates steering + throttle commands

🛰️ Waypoint Planner — Builds navigation graph from GPS coordinates

📡 MQTT Communication — Lightweight distributed system

🌐 Web Dashboard — Live video feed + telemetry

🧱 Modular Codebase — Rover, VLM, UI, and Planner run independently

📁 Project Structure
.
├── cords.txt          # Waypoint list (lat lon alt)
├── rover.py           # Publishes camera frames, executes motor commands
├── gemini.py          # VLM autopilot using Google Gemini API
├── planner.py         # Graph builder + waypoint planner
├── web_ui.py          # Flask UI for live video & telemetry
├── requirements.txt   # Clean Python dependencies
└── README.md

🚀 Quick Start (Simulation on PC)
1️⃣ Create a virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2️⃣ Start the Web UI (live feed + telemetry)

Runs on http://localhost:5000

python web_ui.py

3️⃣ Start the Rover pipeline (simulated camera)
python rover.py

4️⃣ Start the Gemini VLM Autopilot

Processes incoming frames → sends steering commands:

python gemini.py

🛰️ Running on Jetson Nano (Real Rover Mode)

On Jetson Nano (robot):

python3 planner.py
python3 rover.py


On your PC (VLM + dashboard):

python gemini.py
python web_ui.py

📌 Waypoint Format (cords.txt)

Each line must be:

latitude longitude altitude


Example:

17.3970873 78.4897846 500.5


Used by planner.py to compute waypoint graph edges.

📦 Requirements
Flask==3.1.0
paho-mqtt==2.1.0
opencv-python==4.10.0.84
numpy==1.26.0
Pillow==11.2.1
google-generativeai==0.8.5
ultralytics==8.3.29
rplidar==0.10.0
dronekit==2.9.2
pyserial==3.5


These support camera streaming, LiDAR, autonomous navigation, motor control, and the VLM inference.

🔐 API Key Notice

gemini.py uses a Google Gemini API Key.

⚠️ Never commit the API key to GitHub.
Use environment variables or a config.json (excluded via .gitignore).

🛠️ Notes

If LiDAR, DroneKit, or Serial controllers are unavailable, the system falls back to simulation mode

MQTT topics used:

video/stream
rover/cmd
rover/status
rover/obstacle
rover/waypoints
rover/telemetry

📈 Future Improvements

YOLO + LiDAR fusion for obstacle avoidance

Onboard LLM/VLM model (Qwen-VL, Phi-3-Vision, LLaVA)

Improved planner (A*, RRT*, hybrid A*)

ROS2 integration

Web dashboard controls (joystick, live map, logs)