
import cv2
import paho.mqtt.client as mqtt
import base64
import time

BROKER = "192.168.160.213"   # subscriber PC (broker)
PORT = 1883
TOPIC = "/G503/rover/video/stream"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame to JPEG
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # reduce quality for speed
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    # Publish
    client.publish(TOPIC, img_b64)

    # Small delay to control FPS (adjust as needed)
    time.sleep(0.1)

cap.release()
client.disconnect()