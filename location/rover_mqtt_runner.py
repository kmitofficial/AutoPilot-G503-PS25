# rover_client/rover_mqtt_runner.py
import os
import json
import time
import math
import threading
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# reuse your existing code for movement, sensors, and the GeminiRoverBrain if you want LLM locally
from rover_client.gemini import GeminiRoverBrain  # keep as-is (it wraps provider)
from rover_client.rover_runner import get_local_gps, get_rover_heading, send_drive_command, detect_obstacle, haversine_m, bearing_deg, move_rover_towards

load_dotenv()
BROKER = os.getenv("BROKER", "10.208.218.149")
PORT = int(os.getenv("BROKER_PORT", "1883"))
TOPIC_ROUTE = os.getenv("TOPIC_ROUTE", "rover/route")
TOPIC_LOCATION = os.getenv("TOPIC_SUB", "rover/location")  # published by rover
TOPIC_CMD = os.getenv("TOPIC_CMD", "rover/commands")

client = mqtt.Client()
route_lock = threading.Lock()
current_route = []
route_version_ts = None

def on_connect(client, userdata, flags, rc):
    print("[ROVER MQTT] connected", rc)
    client.subscribe(TOPIC_ROUTE)

def on_message(client, userdata, msg):
    global current_route, route_version_ts
    try:
        data = json.loads(msg.payload.decode())
        wps = data.get("waypoints", [])
        if wps:
            with route_lock:
                current_route = wps
                route_version_ts = time.time()
            print(f"[ROVER MQTT] Received route with {len(wps)} waypoints")
    except Exception as e:
        print("[ROVER MQTT] message error", e)

client.on_connect = on_connect
client.on_message = on_message

def publish_location_loop(publish_interval=1.0):
    while True:
        lat, lon = get_local_gps()
        payload = {"lat": lat, "lon": lon}
        try:
            client.publish(TOPIC_LOCATION, json.dumps(payload))
        except Exception as e:
            print("publish location failed:", e)
        time.sleep(publish_interval)

def start_rover_mqtt():
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    threading.Thread(target=publish_location_loop, daemon=True).start()
    print("[ROVER MQTT] started")

def local_follow_loop():
    brain = GeminiRoverBrain()  # optional local LLM; keep cautious with keys
    last_route = None
    while True:
        with route_lock:
            route_copy = list(current_route)
        if not route_copy:
            time.sleep(0.5)
            continue
        # follow route using existing follow_waypoints from rover_runner (but adapted to not fetch routes)
        try:
            from rover_client.rover_runner import follow_waypoints
            follow_waypoints(route_copy, brain)
            # after completion clear route to avoid repeating
            with route_lock:
                current_route.clear()
        except Exception as e:
            print("Error while following route:", e)
            time.sleep(0.5)

if __name__ == "__main__":
    start_rover_mqtt()
    local_follow_loop()
