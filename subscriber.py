
import cv2
import paho.mqtt.client as mqtt
import base64
import numpy as np

BROKER = "192.168.160.213"
PORT = 1883
TOPIC = "/G503/rover/video/stream"

def on_message(client, userdata, msg):
    img_b64 = msg.payload.decode("utf-8")
    img_data = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            client.disconnect()
            cv2.destroyAllWindows()

client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, PORT, 60)

client.subscribe(TOPIC)
print("Waiting for live video stream... Press 'q' to quit.")
client.loop_forever()