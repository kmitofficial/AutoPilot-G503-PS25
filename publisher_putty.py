#rover with cmds
import cv2
import paho.mqtt.client as mqtt
import base64
import time

# ====================
# MQTT Config
# ====================
BROKER = "10.242.57.104"   # Subscriber PC IP
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_CMD = "rover/cmd"    # PC → Rover commands


# ====================
# MQTT Callbacks
# ====================
def on_connect(client, userdata, flags, rc):
    print("✅ Connected to broker")
    client.subscribe(TOPIC_CMD)   # listen for commands


def on_message(client, userdata, msg):
    cmd = msg.payload.decode()
    print(f"📥 Received Command → {cmd}")


# ====================
# Setup MQTT
# ====================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)


# ====================
# Capture Camera & Publish Frames
# ====================
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

frames_to_send = 20   # send for 20 seconds (1 FPS → 20 frames)

for i in range(frames_to_send):
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Encode frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    # Publish frame
    client.publish(TOPIC_VIDEO, img_b64)
    print(f"📤 Sent frame {i+1}/{frames_to_send}")

    # Allow time for incoming commands
    client.loop(timeout=0.1)

    time.sleep(1.0)  # 1 FPS

cap.release()
client.disconnect()