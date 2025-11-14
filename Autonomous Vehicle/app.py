#!/usr/bin/env python3
import base64
import threading
import time
import os

from flask import Flask, render_template, Response
import paho.mqtt.client as mqtt

# --- Configuration (match your existing project settings) ---
BROKER = "10.208.218.104"
PORT = 1883
TOPIC_VIDEO = "video/stream"

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global frame store (latest JPEG bytes)
latest_frame = None
frame_lock = threading.Lock()


def mqtt_on_connect(client, userdata, flags, rc):
    print("[Web-MQTT] Connected with rc=", rc)
    client.subscribe(TOPIC_VIDEO)


def mqtt_on_message(client, userdata, msg):
    global latest_frame
    try:
        # Payload in r1.py is base64.b64encode(jpeg_bytes)
        data = msg.payload
        # decode base64 to get raw jpeg bytes
        jpeg = base64.b64decode(data)
        with frame_lock:
            latest_frame = jpeg
    except Exception as e:
        print("[Web-MQTT] Failed to decode frame:", e)


def start_mqtt():
    client = mqtt.Client()
    client.on_connect = mqtt_on_connect
    client.on_message = mqtt_on_message
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    print("[Web-MQTT] MQTT loop started")
    return client


def generate_mjpeg():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            # no frame yet: yield a small wait and continue
            time.sleep(0.05)
            continue

        # multipart response for MJPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        # rate limit a bit
        time.sleep(0.03)


@app.route("/")
def index():
    # simple page that shows the live feed
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # start mqtt subscriber thread
    mqtt_client = start_mqtt()

    # ensure templates/static exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)

    # start flask
    print("[Web] Starting Flask on http://10.208.218.104:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
