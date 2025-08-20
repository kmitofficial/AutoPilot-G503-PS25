#subscriber+prediction
# pc_predict_and_publish_cmds.py
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import base64
import io
import os
import re
import json
import time
from PIL import Image
import google.generativeai as genai

# =========================
# Config
# =========================
BROKER = "10.242.57.104"     # Your PC (Mosquitto broker) IP
PORT = 1883
TOPIC_VIDEO = "video/stream" # Rover publishes frames here
TOPIC_CMD = "rover/cmd"      # We will publish low-level commands here

SAVE_DIR = "motion_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

API_KEY = "AIzaSyA3Tt8lJNFj6glWaRKIkr_MM3LGvd3gkLw"  # <-- put your key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
print("✅ Gemini model ready")

# =========================
# Helpers
# =========================
prev_frame = None
frame_count = 0

def decode_image(base64_bytes):
    """Decode base64 -> OpenCV frame"""
    img_bytes = base64.b64decode(base64_bytes)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_motion(frame, prev_frame, threshold=5000):
    """Return True if movement is detected."""
    if prev_frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresh)
    return non_zero_count > threshold

def get_prediction_from_gemini(frame):
    """Send frame to Gemini and get prediction text."""
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_bytes = buffer.tobytes()
    prompt = """You are an autonomous driving assistant.
Look at the driving scene and provide:
1. Scene description
2. High-level driving intent
3. Low-level command (speed in m/s and steering angle in degrees)
Format your answer clearly in three sections."""
    resp = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
    return resp.text or ""

def parse_low_level_cmds(text):
    """
    Extract speed (m/s) and steering angle (deg) from Gemini text.
    Robust to different formats. Returns (speed, angle) as floats.
    """
    # Try to find "Speed: X m/s" and "Steering angle: Y degrees"
    speed = None
    angle = None

    # Speed
    m = re.search(r"speed\s*[:\-]?\s*([\-+]?\d+(\.\d+)?)\s*(m/s|mps)?", text, re.I)
    if m:
        speed = float(m.group(1))

    # Steering angle
    a = re.search(r"(steer|steering)\s*angle\s*[:\-]?\s*([\-+]?\d+(\.\d+)?)\s*(deg|degree|degrees)?", text, re.I)
    if a:
        angle = float(a.group(2))

    # Fallbacks: sometimes bullets like "* Speed 2.0" or "Speed = 2"
    if speed is None:
        m2 = re.search(r"\bSpeed\b.*?([\-+]?\d+(\.\d+)?)", text, re.I|re.S)
        if m2: speed = float(m2.group(1))
    if angle is None:
        a2 = re.search(r"\bSteer(?:ing)?\b.*?([\-+]?\d+(\.\d+)?)", text, re.I|re.S)
        if a2: angle = float(a2.group(1))

    # Defaults if not found
    if speed is None: speed = 0.0
    if angle is None: angle = 0.0

    # Clip/clean (you can tune these)
    speed = float(np.clip(speed, -5.0, 25.0))     # m/s
    angle = float(np.clip(angle, -35.0, 35.0))    # deg

    return speed, angle

def publish_cmd(mqtt_client, speed, angle):
    """
    Publish low-level command JSON to rover/cmd.
    Schema:
    {
      "timestamp": <epoch_ms>,
      "speed_mps": <float>,
      "steering_deg": <float>,
      "source": "gemini"
    }
    """
    payload = {
        "timestamp": int(time.time() * 1000),
        "speed_mps": float(speed),
        "steering_deg": float(angle),
        "source": "gemini"
    }
    mqtt_client.publish(TOPIC_CMD, json.dumps(payload), qos=1)
    print(f"📡 Sent CMD → speed={speed:.2f} m/s, steering={angle:.2f}°")

# =========================
# MQTT Handlers
# =========================
def on_connect(client, userdata, flags, rc):
    print("✅ Connected to broker:", rc)
    client.subscribe(TOPIC_VIDEO, qos=0)

def on_message(client, userdata, msg):
    global prev_frame, frame_count
    try:
        frame = decode_image(msg.payload)
        # Show live
        cv2.imshow("Rover Feed", frame)

        # Only run Gemini when motion
        if detect_motion(frame, prev_frame):
            frame_count += 1
            prev_frame = frame.copy()

            # Save frame for debug
            out_path = os.path.join(SAVE_DIR, f"motion_{frame_count}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"\n⚡ Motion detected → Saved {out_path}")

            # Predict
            pred_text = get_prediction_from_gemini(frame)
            print("\n--- Gemini Prediction ---")
            print(pred_text)

            # Parse low-level commands
            speed, angle = parse_low_level_cmds(pred_text)
            # Publish back to rover
            publish_cmd(client, speed, angle)

        # Quit viewer
        if cv2.waitKey(1) & 0xFF == ord('q'):
            client.disconnect()
            cv2.destroyAllWindows()

    except Exception as e:
        print("❌ Error:", e)

# =========================
# Main
# =========================
def main():
    client = mqtt.Client()
    # keep only latest incoming video frame (avoid lag)
    client.max_inflight_messages_set(1)
    client.max_queued_messages_set(1)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if _name_ == "_main_":
    main()