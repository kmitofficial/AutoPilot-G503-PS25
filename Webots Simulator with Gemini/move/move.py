#!/usr/bin/env python3
"""
rover_controller.py
- Sends camera frames every 3s to topic "video/stream"
- Sends GPS every 3s to topic "rover/gps"
- Subscribes to "rover/cmd" and prints commands and executes them on motors
- Publishes obstacle status to "rover/obstacle" (placeholder, can be used later)
"""
import time
import json
import base64
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from controller import Robot

# ---------- CONFIG ----------
BROKER = "127.0.0.1"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_GPS = "rover/gps"
TOPIC_CMD = "rover/cmd"
TOPIC_OBS = "rover/obstacle"

FRAME_INTERVAL = 3.0
GPS_INTERVAL = 3.0
MAX_SPEED = 5.24  # motor maximum velocity in your world

# ---------- GLOBALS ----------
speed_left = 0.0
speed_right = 0.0
executing = False
last_cmd_time = time.time()

# ---------- MQTT callbacks ----------
def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected rc=", rc)
    client.subscribe(TOPIC_CMD)

def on_message(client, userdata, msg):
    global speed_left, speed_right, executing, last_cmd_time
    try:
        data = json.loads(msg.payload.decode())
        speed_left = float(data.get("speed_left", 0.0))
        speed_right = float(data.get("speed_right", 0.0))
        duration = float(data.get("duration_s", 0.0)) if data.get("duration_s") is not None else 0.0
        distance = float(data.get("distance_m", 0.0)) if data.get("distance_m") is not None else 0.0
        executing = (abs(speed_left) > 0.001 or abs(speed_right) > 0.001)
        last_cmd_time = time.time()

        # Print what we received from Gemini
        print(f"[GEMINI CMD] L={speed_left:.2f}% R={speed_right:.2f}% Dist={distance:.2f}m Dur={duration:.2f}s")

    except Exception as e:
        print("[CMD Parse Error]", e)

# ---------- Helper functions ----------
def encode_frame_to_jpeg_bytes(np_img):
    ret, buf = cv2.imencode(".jpg", np_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        return None
    return base64.b64encode(buf)

# ---------- Main ----------
def main():
    global executing

    # Webots setup
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Devices - check exact names in your scene tree and update if different
    left_motor = robot.getDevice("left wheel")
    right_motor = robot.getDevice("right wheel")
    camera = robot.getDevice("camera")
    gps = robot.getDevice("gps")

    if not left_motor or not right_motor:
        print("[ERROR] Wheel motors not found! Check device names in Webots.")
        return

    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    camera.enable(timestep)
    gps.enable(timestep)

    # MQTT setup
    mqttc = mqtt.Client()
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    try:
        mqttc.connect(BROKER, PORT, 60)
    except Exception as e:
        print("[MQTT] Could not connect to broker:", e)
        mqttc = None

    if mqttc:
        mqttc.loop_start()

    last_cam_send = 0.0
    last_gps_send = 0.0

    try:
        while robot.step(timestep) != -1:
            now = time.time()

            # Camera frame publish every FRAME_INTERVAL
            if mqttc and (now - last_cam_send >= FRAME_INTERVAL):
                img = camera.getImage()
                if img:
                    w = camera.getWidth()
                    h = camera.getHeight()
                    np_img = np.frombuffer(img, np.uint8).reshape((h, w, 4))
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2BGR)
                    payload = encode_frame_to_jpeg_bytes(np_img)
                    if payload:
                        mqttc.publish(TOPIC_VIDEO, payload)
                        # small console feedback
                        print("[CAMERA] Frame published")
                last_cam_send = now

            # GPS publish every GPS_INTERVAL
            if mqttc and (now - last_gps_send >= GPS_INTERVAL):
                g = gps.getValues()
                payload = json.dumps({
                    "timestamp": int(now*1000),
                    "x": g[0],
                    "y": g[1],
                    "z": g[2]
                })
                mqttc.publish(TOPIC_GPS, payload)
                print(f"[GPS] x={g[0]:.2f} y={g[1]:.2f} z={g[2]:.2f}")
                last_gps_send = now

            # Execute movement based on latest Gemini command percentages
            if executing:
                # gemini sends percent PWM [-100..100], convert to velocity
                l_vel = np.clip(speed_left / 100.0 * MAX_SPEED, -MAX_SPEED, MAX_SPEED)
                r_vel = np.clip(speed_right / 100.0 * MAX_SPEED, -MAX_SPEED, MAX_SPEED)
                left_motor.setVelocity(l_vel)
                right_motor.setVelocity(r_vel)
            else:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

            # Safety timeout
            if time.time() - last_cmd_time > 10.0:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                executing = False

    except KeyboardInterrupt:
        print("[INFO] Manual stop")
    finally:
        if mqttc:
            mqttc.loop_stop()
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        print("[INFO] Controller exiting")

if __name__ == "__main__":
    main()
