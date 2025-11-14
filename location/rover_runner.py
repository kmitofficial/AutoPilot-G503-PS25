# ==========================================================
# rover_client/rover_runner.py — LOCAL NAVIGATION (Option B)
# Receives route via MQTT: topic = rover/route
# Sends GPS to MQTT: topic = rover/location
# ==========================================================

import time
import math
import json
import os
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from rover_client.gemini import GeminiRoverBrain

# ======================================================
# ENV + MQTT
# ======================================================
load_dotenv()
BROKER = os.getenv("BROKER", "10.208.218.149")
PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC_ROUTE = "rover/route"       # server → rover
TOPIC_LOCATION = "rover/location" # rover → server

client = mqtt.Client()

# Live route buffer
CURRENT_ROUTE = []
ROUTE_TS = None

# ======================================================
# Simulated Rover Sensors (your existing code)
# ======================================================
_start_lat = 17.397048
_start_lon = 78.489775
_fake_step = 0

def detect_obstacle():
    import random
    if random.random() < 0.05:
        return True, round(random.uniform(0.3,2.5),2), random.choice(["left","front","right"])
    return False, None, None

def get_local_gps():
    global _fake_step, _start_lat, _start_lon
    _fake_step += 1
    lat = _start_lat + (_fake_step * 0.00004)
    lon = _start_lon + (_fake_step * 0.00002)
    return lat, lon

def get_rover_heading():
    return 45.0

def send_drive_command(speed, steering):
    print(f"[DRIVE] speed={speed:.2f} steering={steering:.3f}")

def haversine_m(a,b,c,d):
    R=6371000.0
    φ1,φ2=map(math.radians,(a,c))
    dφ,dl=map(math.radians,(c-a,d-b))
    s=math.sin(dφ/2)**2+math.cos(φ1)*math.cos(φ2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(s))

def bearing_deg(a,b,c,d):
    φ1,φ2=map(math.radians,(a,c))
    dl=math.radians(d-b)
    x=math.sin(dl)*math.cos(φ2)
    y=math.cos(φ1)*math.sin(φ2)-math.sin(φ1)*math.cos(φ2)*math.cos(dl)
    return (math.degrees(math.atan2(x,y))+360)%360

# ======================================================
# FOLLOW WAYPOINTS (unchanged)
# ======================================================
def follow_waypoints(waypoints, brain):
    total_wp=len(waypoints)
    dest_lat,dest_lon=waypoints[-1]

    start_lat,start_lon=get_local_gps()
    brain.start_route(start_lat,start_lon,dest_lat,dest_lon,total_wp)

    for i,target in enumerate(waypoints):
        while True:
            lat,lon=get_local_gps()
            d=haversine_m(lat,lon,target[0],target[1])
            if d<1.0:
                print(f"✅ Reached waypoint {i+1}/{total_wp}")
                break

            desired=bearing_deg(lat,lon,target[0],target[1])
            heading=get_rover_heading()
            err=(desired-heading+540)%360-180

            Kp=0.02
            steering=max(-1.0,min(1.0,Kp*err))
            speed=0.6*(1-min(abs(err)/90.0,0.9))
            if d<3.0: speed*=0.5

            brain.update(lat,lon,target[0],target[1],i,total_wp,d)

            # obstacle
            hit,dist,dirc=detect_obstacle()
            if hit:
                print(f"⚠️ Obstacle {dist}m ({dirc})")
                brain.obstacle_detected(dist,dirc)

                if dist<0.5:
                    send_drive_command(0,0)
                    print("🛑 STOP: close obstacle")
                    time.sleep(1.5)
                    continue
                elif dist<1.5:
                    send_drive_command(0.2,steering)
                    print("⚠ Slow down")
                    time.sleep(0.5)

            send_drive_command(speed,steering)
            time.sleep(0.25)

    send_drive_command(0,0)
    print("🎯 Route complete.")
    brain.finish_route()

# ======================================================
# MQTT CALLBACKS
# ======================================================
def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected.", rc)
    client.subscribe(TOPIC_ROUTE)

def on_message(client, userdata, msg):
    global CURRENT_ROUTE, ROUTE_TS
    try:
        data = json.loads(msg.payload.decode())
        wps = data.get("waypoints", [])
        if wps:
            CURRENT_ROUTE = wps
            ROUTE_TS = time.time()
            print(f"[MQTT] Received new route: {len(wps)} points")
    except Exception as e:
        print("[MQTT] ERROR parsing route:", e)

client.on_connect = on_connect
client.on_message = on_message

# ======================================================
# GPS publisher thread
# ======================================================
import threading
def gps_publisher():
    while True:
        lat,lon=get_local_gps()
        payload={"lat":lat,"lon":lon}
        client.publish(TOPIC_LOCATION, json.dumps(payload))
        time.sleep(1.0)

# ======================================================
# MAIN LOOP
# ======================================================
def main_loop():
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    threading.Thread(target=gps_publisher, daemon=True).start()
    brain = GeminiRoverBrain()

    last_route=[]

    while True:
        if CURRENT_ROUTE and CURRENT_ROUTE != last_route:
            print("🗺️ Starting route...")
            follow_waypoints(CURRENT_ROUTE, brain)
            last_route = CURRENT_ROUTE[:]
        time.sleep(0.3)

# ======================================================
if __name__ == "__main__":
    main_loop()
