# backend_server/mqtt_bridge.py
import os
import json
import paho.mqtt.client as mqtt
from backend_server.rover_location_provider import update_rover_location

# ======================================================
# ENVIRONMENT
# ======================================================
BROKER = os.getenv("BROKER", "10.208.218.149")
PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC_LOCATION = os.getenv("TOPIC_LOCATION", "rover/location")
TOPIC_CMD = os.getenv("TOPIC_CMD", "rover/commands")
TOPIC_ROUTE = os.getenv("TOPIC_ROUTE", "rover/route")

# ======================================================
# MQTT CLIENT
# ======================================================
client = mqtt.Client()


# ======================================================
# CONNECTION HANDLER
# ======================================================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected rc={rc}")

    # Subscribe to rover/location updates
    print(f"[MQTT] Subscribing to {TOPIC_LOCATION}")
    client.subscribe(TOPIC_LOCATION)


# ======================================================
# MESSAGE HANDLER
# ======================================================
def on_message(client, userdata, msg):
    """
    Handles incoming MQTT messages.
    Updates rover location on 'rover/location'.
    """
    try:
        raw = msg.payload.decode()
        print(f"[MQTT] Incoming → Topic: {msg.topic} | Raw: {raw}")

        data = json.loads(raw)

        if msg.topic == TOPIC_LOCATION:
            lat = data.get("lat")
            lon = data.get("lon")

            print(f"[MQTT] Parsed → lat={lat}, lon={lon}")

            if lat is not None and lon is not None:
                update_rover_location(lat, lon)
                print("[MQTT] ✔ Rover location UPDATED")

    except Exception as e:
        print("[MQTT] Error processing message:", e)


client.on_connect = on_connect
client.on_message = on_message


# ======================================================
# MQTT STARTUP
# ======================================================
def start_mqtt():
    try:
        print(f"[MQTT] Connecting to {BROKER}:{PORT}")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        print(f"[MQTT] ✔ Bridge running at {BROKER}:{PORT}")
    except Exception as e:
        print("[MQTT] Connection error:", e)


# ======================================================
# COMMAND PUBLISHER
# ======================================================
def send_command(speed, steering):
    payload = {"speed": float(speed), "steering": float(steering)}
    client.publish(TOPIC_CMD, json.dumps(payload))
    print(f"[MQTT] → Sent command: {payload}")


# ======================================================
# ROUTE PUBLISHER
# ======================================================
def send_route(waypoints):
    """
    Publishes the route (list of [lat, lon]) to rover.
    """
    payload = {"waypoints": waypoints}
    client.publish(TOPIC_ROUTE, json.dumps(payload))
    print(f"[MQTT] → Published route ({len(waypoints)} waypoints) → {TOPIC_ROUTE}")

