# ==========================================================
# rover_client/rover_runner.py
# ==========================================================
# Autonomous Rover Navigation Client
# - Fetches route from Flask server
# - Simulates GPS + Obstacle detection
# - Uses Gemini/LLM reasoning for decisions
# ==========================================================

import time
import math
import requests
import os
from dotenv import load_dotenv
# from rover_client.gemini import GeminiRoverBrain  # ✅ Fixed import

load_dotenv()
SERVER = os.getenv("SERVER_URL", "http://localhost:8000")

# ======================================================
# 🛰️ Rover State Simulation (Replace with real sensors)
# ======================================================
ROVER_POSITION = [17.397048, 78.489775]  # default starting point
ROVER_HEADING = 0.0  # degrees (North)
STEP_DISTANCE = 0.00005  # ~5m per update for testing

def get_local_gps():
    """Simulate rover GPS position."""
    return tuple(ROVER_POSITION)

def get_rover_heading():
    """Return heading from IMU (static simulation)."""
    global ROVER_HEADING
    return ROVER_HEADING

def move_rover_towards(lat, lon):
    """Simulate rover moving toward target coordinates."""
    global ROVER_POSITION
    clat, clon = ROVER_POSITION
    dlat = lat - clat
    dlon = lon - clon
    dist = math.sqrt(dlat**2 + dlon**2)
    if dist < STEP_DISTANCE:
        ROVER_POSITION = [lat, lon]
    else:
        ROVER_POSITION = [
            clat + (dlat / dist) * STEP_DISTANCE,
            clon + (dlon / dist) * STEP_DISTANCE,
        ]

def send_drive_command(speed_m_s, steering):
    """Placeholder for motor control command."""
    print(f"[DRIVE] speed={speed_m_s:.2f} m/s steering={steering:.3f}")

# ======================================================
# 🧱 Obstacle Detection Simulation
# ======================================================
def detect_obstacle():
    """
    Simulated obstacle detection.
    Replace with real ultrasonic or LiDAR sensor data.
    """
    import random
    if random.random() < 0.05:  # 5% chance of obstacle per iteration
        return True, round(random.uniform(0.4, 2.5), 2), random.choice(["left", "front", "right"])
    return False, None, None

# ======================================================
# 🧮 Math Utilities
# ======================================================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi, dl = map(math.radians, (lat2 - lat1, lon2 - lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl)*math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

# ======================================================
# 🤖 Waypoint Navigation
# ======================================================
def follow_waypoints(waypoints, brain):
    if not waypoints:
        print("⚠️ No route received from server.")
        return

    total_wp = len(waypoints)
    dest_lat, dest_lon = waypoints[-1]

    start_lat, start_lon = get_local_gps()
    brain.start_route(start_lat, start_lon, dest_lat, dest_lon, total_wp)

    for i, target in enumerate(waypoints):
        target_lat, target_lon = target
        while True:
            lat, lon = get_local_gps()
            move_rover_towards(target_lat, target_lon)
            d = haversine_m(lat, lon, target_lat, target_lon)

            if d < 1.0:
                print(f"✅ Reached waypoint {i+1}/{total_wp}")
                break

            desired = bearing_deg(lat, lon, target_lat, target_lon)
            heading = get_rover_heading()
            err = (desired - heading + 540) % 360 - 180

            # proportional steering control
            Kp = 0.02
            steering = max(-1.0, min(1.0, Kp * err))
            max_speed = 0.6
            speed = max_speed * (1 - min(abs(err)/90.0, 0.9))
            if d < 3.0:
                speed *= 0.5

            # update reasoning
            brain.update(lat, lon, target_lat, target_lon, i, total_wp, d)

            # obstacle handling
            detected, dist, direction = detect_obstacle()
            if detected:
                print(f"⚠️ Obstacle detected {dist}m ({direction})")
                brain.obstacle_detected(dist, direction)

                if dist < 0.5:
                    send_drive_command(0, 0)
                    print("🛑 Stopped (close obstacle)")
                    time.sleep(1.5)
                    continue
                elif dist < 1.5:
                    send_drive_command(0.2, steering)
                    print("⚠️ Slowing down")
                    time.sleep(0.5)

            send_drive_command(speed, steering)
            time.sleep(0.25)

    send_drive_command(0, 0)
    print("🎯 Destination reached.")
    brain.finish_route()

# ======================================================
# 🧭 Main Loop
# ======================================================
def main_loop(poll_interval=1.5):
    brain = GeminiRoverBrain()
    last_waypoints = []
    print("🚀 Rover Runner active. Awaiting route from server...")

    while True:
        # report current GPS
        lat, lon = get_local_gps()
        try:
            requests.post(f"{SERVER}/report_location", json={"lat": lat, "lon": lon}, timeout=5)
        except Exception as e:
            print("⚠️ Failed to report location:", e)

        # fetch route
        try:
            r = requests.get(f"{SERVER}/get_route", timeout=5).json()
            waypoints = r.get("waypoints", [])
        except Exception as e:
            print("⚠️ Failed to get route:", e)
            waypoints = []

        # follow new route if changed
        if waypoints and waypoints != last_waypoints:
            print(f"🗺️  New route received with {len(waypoints)} waypoints.")
            follow_waypoints(waypoints, brain)
            last_waypoints = waypoints

        time.sleep(poll_interval)
# ==========================================================
# rover_client/gemini.py
# ==========================================================
# Unified Gemini + OpenAI reasoning brain for autonomous rover
# ==========================================================

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# 🔑 Environment
# ==========================================================
SERVER = os.getenv("SERVER_URL", "http://localhost:8000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Detect which model to use
if GEMINI_API_KEY:
    PROVIDER = "gemini"
elif OPENAI_API_KEY:
    PROVIDER = "openai"
else:
    PROVIDER = "local"

print(f"[GeminiRoverBrain] Using provider: {PROVIDER.upper()}")

# ==========================================================
# 🤖 LLM Providers
# ==========================================================
if PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    def _generate(prompt: str) -> str:
        """Generate reasoning from Google Gemini."""
        try:
            model = genai.GenerativeModel("gemini-flash-latest")
            response = model.generate_content(prompt)
            if not response or not response.text:
                return "[Gemini] Empty response from model."
            return response.text.strip()
        except Exception as e:
            return f"[Gemini Error] {str(e)}"



elif PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    def _generate(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {e}"

else:
    # Fallback when offline or no API key
    def _generate(prompt: str) -> str:
        return f"(Offline reasoning fallback) {prompt[:100]}..."

# ==========================================================
# 🌐 Log reasoning to server dashboard
# ==========================================================
def log_to_server(message: str):
    try:
        requests.post(f"{SERVER}/log_gemini", json={"out": message}, timeout=3)
    except Exception as e:
        print("⚠️ Failed to log to server:", e)

# ==========================================================
# 🧠 Rover reasoning functions
# ==========================================================
def think(prompt: str) -> str:
    out = _generate(prompt)
    log_to_server(out)
    print(f"[Gemini] {out}")
    return out


def describe_start(lat, lon, dest_lat, dest_lon, total_points):
    return think(
        f"Rover starting mission from ({lat:.5f}, {lon:.5f}) to ({dest_lat:.5f}, {dest_lon:.5f}) "
        f"with {total_points} waypoints. Provide a concise mission plan and safety note."
    )


def reason_during_drive(cur_lat, cur_lon, target_lat, target_lon, wp_index, total_wp, distance_m):
    return think(
        f"Rover update: Position=({cur_lat:.5f}, {cur_lon:.5f}), "
        f"Next waypoint {wp_index+1}/{total_wp} ({target_lat:.5f}, {target_lon:.5f}), "
        f"Distance={distance_m:.1f}m. Suggest speed/steering adjustment."
    )


def handle_obstacle(distance_m, direction="front"):
    return think(
        f"Obstacle detected {distance_m:.2f}m ahead ({direction}). Recommend safe maneuver "
        f"(stop, slow, or steer away)."
    )


def describe_completion():
    return think("Rover reached destination safely. Summarize navigation and safety actions.")


# ==========================================================
# 🧭 Rover Intelligence Brain
# ==========================================================
class GeminiRoverBrain:
    def __init__(self):
        self.route_started = False
        self.route_completed = False
        print(f"✅ GeminiRoverBrain initialized with {PROVIDER.upper()} provider.")

    def start_route(self, lat, lon, dest_lat, dest_lon, total_points):
        self.route_started = True
        return describe_start(lat, lon, dest_lat, dest_lon, total_points)

    def update(self, cur_lat, cur_lon, target_lat, target_lon, wp_index, total_wp, distance_m):
        return reason_during_drive(cur_lat, cur_lon, target_lat, target_lon, wp_index, total_wp, distance_m)

    def finish_route(self):
        self.route_completed = True
        return describe_completion()

    def obstacle_detected(self, distance_m, direction="front"):
        return handle_obstacle(distance_m, direction)

# ==========================================================
# 🧪 Self-Test
# ==========================================================
if __name__ == "__main__":
    print("🧠 GeminiRoverBrain Self-Test")
    brain = GeminiRoverBrain()
    brain.start_route(17.3850, 78.4867, 17.4400, 78.4999, 25)
    time.sleep(1)
    brain.update(17.3900, 78.4900, 17.3950, 78.4950, 1, 25, 12.3)
    time.sleep(1)
    brain.finish_route()

# ======================================================
# 🚀 Entry Point
# ======================================================
if __name__ == "__main__":
    main_loop()
