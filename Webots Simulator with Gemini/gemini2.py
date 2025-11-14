#!/usr/bin/env python3
"""
gemini.py
- Subscribes to "video/stream" (JPEG base64) and "rover/gps"
- Runs VLM (Gemini) produces commands, publishes "rover/cmd"
- Hosts a Flask dashboard at http://127.0.0.1:5000 with live MJPEG and status panels
- Auto-opens the dashboard in a browser when started
"""
import os
import io
import re
import time
import json
import base64
import threading
import webbrowser
from PIL import Image
from flask import Flask, Response, request, jsonify, render_template_string
import paho.mqtt.client as mqtt
import numpy as np

# optional: use the GenAI SDK if available
try:
    import google.generativeai as genai
    USE_GENAI = True
except Exception:
    USE_GENAI = False

# ---------- CONFIG ----------
BROKER = "127.0.0.1"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_CMD = "rover/cmd"
TOPIC_GPS = "rover/gps"
TOPIC_OBS = "rover/obstacle"

SAVE_DIR = "motion_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

#API_KEY = ""  # set if using genai
if USE_GENAI:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

MIN_INTERVAL = 0.8
MAX_CMD_DURATION = 1.0

# ---------- Shared state ----------
latest_frame_jpeg = None          # raw JPEG bytes (not base64)
latest_gps = {"x": 0.0, "y": 0.0, "z": 0.0}
scene_text = ""
high_level_intent = ""
low_level_cmd = {"speed_left": 0.0, "speed_right": 0.0, "distance_m": 0.0, "duration_s": 0.0}
destination = None                # user-provided destination {x,y,z}
last_model_time = 0.0
last_sent = 0.0
rover_reports_obstacle = False

# ---------- Flask app (dashboard) ----------
app = Flask(__name__)

DASH_HTML = """
<!doctype html>
<html>
<head>
  <title>Rover Dashboard</title>
  <meta charset="utf-8">
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 12px; background:#f6f8fb; }
    #container { max-width:1100px; margin: 0 auto; }
    .video { width:100%; height:420px; background:#333; display:block; border-radius:6px; object-fit:cover; }
    .panel { background:#3b71c1; color:white; padding:14px; margin-top:12px; border-radius:6px; text-align:center; font-weight:600;}
    .small { background:#2f5496; padding:10px; margin-top:8px; border-radius:6px; color:#fff; }
    .row { display:flex; gap:12px; margin-top:12px; }
    .col { flex:1; }
    input[type="text"]{width:100%;padding:8px;border-radius:4px;border:1px solid #ccc;}
    button{padding:8px 12px;border:none;border-radius:4px;background:#2b66b4;color:#fff;cursor:pointer;}
    .info { background:#fff;padding:12px;border-radius:6px;margin-top:8px; color:#111; box-shadow: 0 2px 6px rgba(0,0,0,0.08)}
  </style>
</head>
<body>
  <div id="container">
    <h2>PICK A LOCATION AS DESTINATION (GPS)</h2>
    <img id="video" class="video" src="/video_feed">
    <div class="panel">SCENE DESCRIPTION</div>
    <div class="info" id="scene">loading...</div>
    <div class="panel">High level intent</div>
    <div class="info" id="intent">loading...</div>
    <div class="panel">Low level commands</div>
    <div class="info" id="commands">loading...</div>

    <div class="row">
      <div class="col">
        <div style="margin-top:12px;">
          <label>Destination (x,y,z) — comma separated</label>
          <input id="dest_input" type="text" placeholder="e.g. 1.0,0.0,2.5">
        </div>
      </div>
      <div style="width:150px;align-self:end">
        <button onclick="setDest()">Set Destination</button>
      </div>
    </div>

    <div class="row">
      <div class="col">
        <div class="info" id="gps">GPS: loading...</div>
      </div>
    </div>
  </div>

<script>
async function refresh() {
  try {
    const r = await fetch('/status');
    const j = await r.json();
    document.getElementById('scene').innerText = j.scene_text || '—';
    document.getElementById('intent').innerText = j.high_level_intent || '—';
    document.getElementById('commands').innerText = JSON.stringify(j.low_level_cmd);
    document.getElementById('gps').innerText = 'GPS: ' + (j.gps ? (j.gps.x.toFixed(2)+', '+j.gps.y.toFixed(2)+', '+j.gps.z.toFixed(2)) : '—');
  } catch(e) {
    console.log('status error', e);
  }
}
function setDest(){
  const v = document.getElementById('dest_input').value.trim();
  if(!v) return alert('enter x,y,z');
  fetch('/set_destination', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({dest:v})})
    .then(()=> alert('destination set'))
    .catch(e=> alert('error'));
}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASH_HTML)

@app.route("/status")
def status():
    return jsonify({
        "scene_text": scene_text,
        "high_level_intent": high_level_intent,
        "low_level_cmd": low_level_cmd,
        "gps": latest_gps,
        "destination": destination
    })

@app.route("/set_destination", methods=["POST"])
def set_destination():
    global destination
    data = request.get_json()
    if not data:
        return "bad", 400
    dest_raw = data.get("dest") or data.get("destination")
    if not dest_raw:
        return "bad", 400
    try:
        parts = [float(x.strip()) for x in dest_raw.split(",")]
        if len(parts) != 3:
            return "bad format", 400
        destination = {"x": parts[0], "y": parts[1], "z": parts[2]}
        print("[DASH] New destination:", destination)
        return "ok"
    except Exception as e:
        return str(e), 400

def generate_mjpeg():
    """Yield MJPEG frames for <img src="/video_feed">"""
    global latest_frame_jpeg
    while True:
        if latest_frame_jpeg:
            # multipart frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_jpeg + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- MQTT client and VLM loop ----------
def decode_b64_to_jpeg(b64payload):
    try:
        raw = base64.b64decode(b64payload)
        return raw
    except Exception:
        return None

def safe_parse_json(text):
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except:
                return None
    return None

def model_query(frame_bytes, dest):
    """Call Gemini model or a simple heuristic if genai not available.
       frame_bytes: raw jpeg bytes
       dest: destination dict or None
       returns (scene_text, speed_m_s, steering_deg, distance_m)
    """
    global USE_GENAI, model
    if USE_GENAI:
        prompt = """
You are an autonomous driving assistant. Respond only with JSON:
{"scene_description":"...","speed_m_s":<float>,"steering_deg":<float>,"distance_m":<float>}
Rules: prefer speeds 0-3 m/s, steering -35..35 degrees. If destination is provided try to head that way.
"""
        try:
            # attach image & prompt
            result = model.generate_content([prompt, {"mime_type":"image/jpeg","data":frame_bytes}])
            text = result.text or ""
            parsed = safe_parse_json(text)
            if parsed:
                return text, float(parsed.get("speed_m_s", 0.0)), float(parsed.get("steering_deg", 0.0)), float(parsed.get("distance_m", 0.0))
            else:
                return text, 0.0, 0.0, 0.0
        except Exception as e:
            print("[GENAI] error", e)
            return str(e), 0.0, 0.0, 0.0
    else:
        # fallback heuristic: simple forward command unless obstacle flagged
        return "fallback: drive forward", 1.0, 0.0, 2.0

def on_connect(client, userdata, flags, rc):
    print("[MQTT] connected rc=", rc)
    client.subscribe(TOPIC_VIDEO)
    client.subscribe(TOPIC_GPS)
    client.subscribe(TOPIC_OBS)

def on_message(client, userdata, msg):
    global latest_frame_jpeg, latest_gps, scene_text, high_level_intent, low_level_cmd, last_model_time, last_sent, rover_reports_obstacle
    try:
        if msg.topic == TOPIC_VIDEO:
            # payload is base64 encoded jpeg bytes
            latest_frame_jpeg = decode_b64_to_jpeg(msg.payload)
            # store a small snapshot on disk occasionally
            # awaken the model processing loop if enough time passed
            now = time.time()
            if now - last_model_time >= MIN_INTERVAL and not rover_reports_obstacle:
                # run model in another thread to avoid blocking MQTT
                frame_copy = latest_frame_jpeg
                threading.Thread(target=process_frame_and_send_cmd, args=(frame_copy,)).start()
                last_model_time = now

        elif msg.topic == TOPIC_GPS:
            try:
                latest_gps = json.loads(msg.payload.decode())
            except Exception:
                pass

        elif msg.topic == TOPIC_OBS:
            # rover may publish obstacles; use it to pause inference
            try:
                d = json.loads(msg.payload.decode())
                rover_reports_obstacle = bool(d.get("obstacle", False))
            except:
                rover_reports_obstacle = msg.payload.decode().strip().lower() in ("1","true","yes")

    except Exception as e:
        print("[MQTT on_message] error", e)

def process_frame_and_send_cmd(frame_bytes):
    """Run the VLM/model, parse result, publish rover/cmd, and update dashboard state"""
    global scene_text, high_level_intent, low_level_cmd, last_sent, destination

    if not frame_bytes:
        return

    # call model (or fallback)
    txt, s, a, distance = model_query(frame_bytes, destination)
    scene_text = txt[:400]  # truncate for dashboard
    high_level_intent = "navigate to destination" if destination else "local navigation"

    # compute duration and PWM mapping
    duration_s = 0.0
    if s > 0.05 and distance > 0.01:
        duration_s = min(distance / max(s, 0.05), MAX_CMD_DURATION)
    if s <= 0.05:
        l = 0.0; r = 0.0
    else:
        base = (s / 20.0) * 100.0
        diff = (a / 35.0) * 50.0
        l = float(np.clip(base + diff, -100, 100))
        r = float(np.clip(base - diff, -100, 100))

    low_level_cmd = {"speed_left": round(l,2), "speed_right": round(r,2), "distance_m": round(distance,3), "duration_s": round(duration_s,3)}

    # Publish command
    payload = json.dumps(low_level_cmd)
    mqtt.publish(TOPIC_CMD, payload, qos=1)
    last_sent = time.time()
    print("[GEMINI->CMD]", low_level_cmd)

# ---------- MQTT connect with retry ----------
mqtt = mqtt.Client()
mqtt.on_connect = on_connect
mqtt.on_message = on_message

def mqtt_connect_with_retry():
    while True:
        try:
            mqtt.connect(BROKER, PORT, 60)
            print("[MQTT] connected to broker")
            mqtt.loop_start()
            break
        except Exception as e:
            print("[MQTT] broker not reachable, retry in 3s", e)
            time.sleep(3)

# ---------- Start Flask in thread and auto-open browser ----------
def start_flask():
    # Flask runs on port 5000
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)

# ---------- Main ----------
if __name__ == "__main__":
    mqtt_connect_with_retry()
    # start Flask UI
    threading.Thread(target=start_flask, daemon=True).start()
    print("[INFO] Dashboard should open in your default browser")
    # main thread just keeps MQTT alive (Flask and MQTT loop running)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down")
        mqtt.loop_stop()
        os._exit(0)
