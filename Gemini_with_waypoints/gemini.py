import paho.mqtt.client as mqtt
import cv2
import numpy as np
import base64, io, os, re, json, time
from PIL import Image
import google.generativeai as genai
import math

# =========================
# Config
# =========================
BROKER = "YOUR_BROKERip"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_CMD = "rover/cmd"
TOPIC_OBS = "rover/obstacle"
TOPIC_WAYPOINTS = 'rover/waypoints'
TOPIC_TELEMETRY = 'rover/telemetry'
SAVE_DIR = "motion_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# API KEY: hard-code your Gemini API key here.
# WARNING: Hard-coding API keys in source is insecure. Replace the placeholder
# below with your actual key. Do NOT commit the real key to version control.
API_KEY = "YOUR_api_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
print("✅ Gemini model ready (using hard-coded API key)")

prev_frame = None
last_sent = 0.0
frame_count = 0
rover_has_obstacle = False
waypoints = []
telemetry = {}

# backoff state for Gemini quota/rate-limit handling
gemini_backoff_until = 0.0
gemini_backoff_multiplier = 1.0
GEMINI_BACKOFF_MAX_MULT = 16.0
GEMINI_BACKOFF_STEP = 2.0
# last command published (kept so we can republish as a keepalive during backoff)
last_published_cmd = None
last_published_time = 0.0
KEEPALIVE_REPUB_INTERVAL = 2.0
# allow occasional probe attempts while in backoff (seconds)
GEMINI_BACKOFF_PROBE_INTERVAL = 30.0
gemini_last_probe_time = 0.0

# throttle: minimum seconds between model invocations
# throttle: minimum seconds between model invocations
MIN_INTERVAL = 0.15   # seconds (allow faster invocations if frames arrive)

# max duration we allow a command to ask the rover to run (seconds)
MAX_CMD_DURATION = 10.0

# =========================
# Helper Functions
# =========================
def decode_image(base64_bytes):
    img_bytes = base64.b64decode(base64_bytes)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_motion(frame, prev_frame, threshold=1000):
    if prev_frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return np.count_nonzero(mask) > threshold

def get_prediction_from_gemini(frame):
    # downscale image to reduce model latency
    h, w = frame.shape[:2]
    target_w = 640
    if w > target_w:
        nh = int(target_w * h / w)
        small = cv2.resize(frame, (target_w, nh))
    else:
        small = frame
    _, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    img_bytes = buf.tobytes()
    prompt = """
You are an autonomous driving AI assistant.
Respond ONLY in JSON (no extra explanation):
{
    "scene_description": "brief summary",
    "speed_m_s": <float>,
    "steering_deg": <float>,
    "distance_m": <float>,
    "heading_deg": <float>,            # absolute heading estimate (0-360) if available
    "suggested_node": <int or null>,   # optional: index of suggested next map node
    "traffic_light": "none|red|yellow|green", # detected traffic light color
    "stop_on_traffic_light": <true|false>
}
Rules:
- If path clear: speed 1–3 m/s, distance 0.5–3 m
- Turn ± steering_deg if needed (-35 to 35)
- If obstacle ahead: speed 0, distance 0
- If a red traffic light is detected ahead, set stop_on_traffic_light=true and speed 0

IMPORTANT: If an ActiveWaypoint context is provided (see appended context), generate commands that move TOWARD THAT ActiveWaypoint only. Do NOT propose alternate routes or direct the rover to other, unrelated map nodes. The "suggested_node" field may be null or may contain an index that is on the path toward the ActiveWaypoint.
"""
    # If an active waypoint / telemetry is known, append a small context block
    try:
        if telemetry and waypoints:
            # pick nearest waypoint not yet reached
            def hav(lat1, lon1, lat2, lon2):
                R = 6371000.0
                phi1 = math.radians(lat1); phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
                a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            lat0 = float(telemetry.get('lat', 0.0))
            lon0 = float(telemetry.get('lon', 0.0))
            nearest = None; ndist = 1e9
            for w in waypoints:
                try:
                    d = hav(lat0, lon0, float(w.get('lat')), float(w.get('lon')))
                except Exception:
                    continue
                if d < ndist:
                    ndist = d; nearest = w
            if nearest is not None:
                ctx = f"\nActiveWaypoint: lat={nearest.get('lat')} lon={nearest.get('lon')} alt={nearest.get('alt')} dist_to_wp_m={ndist:.2f}\nCurrentTelemetry: lat={lat0} lon={lon0} heading={telemetry.get('heading', 'unknown')}\n"
                prompt = prompt + ctx
    except Exception:
        pass
    # Backoff globals (declare once at top of function before use)
    global gemini_backoff_multiplier, gemini_backoff_until
    # If we're in backoff due to earlier quota errors, normally skip calling Gemini.
    # However, allow a periodic probe so the client can recover automatically when
    # the service quota window resets (avoids permanent silence).
    global gemini_last_probe_time
    now = time.time()
    if now < gemini_backoff_until:
        # allow a probe attempt only if the probe interval has elapsed
        if (now - gemini_last_probe_time) < GEMINI_BACKOFF_PROBE_INTERVAL:
            # skip model call during backoff
            return ""
        # perform a probe attempt
        gemini_last_probe_time = now

    try:
        # send prompt + image
        result = model.generate_content([prompt, {"mime_type":"image/jpeg","data":img_bytes}])
        # success -> reset backoff multiplier
        gemini_backoff_multiplier = 1.0
        gemini_backoff_until = 0.0
        return result.text or ""
    except Exception as e:
        # Handle rate-limit / quota errors gracefully by backing off when possible
        msg = str(e)
        print("[Gemini] Error:", msg)
        retry_sec = None
        m = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)s", msg, re.I)
        if not m:
            m = re.search(r"retry_delay\W*\{[^}]*seconds\W*:\W*([0-9]+)\W*\}", msg, re.I)
        if m:
            try:
                retry_sec = float(m.group(1))
            except Exception:
                retry_sec = None
        if retry_sec is None and 'quota' in msg.lower():
            retry_sec = 15.0
        try:
            delay = float(retry_sec) if retry_sec else 15.0
            delay = delay * gemini_backoff_multiplier
            gemini_backoff_until = time.time() + delay
            print(f"[Gemini] Backing off for {delay:.1f}s (mult={gemini_backoff_multiplier}) due to rate limit / quota")
            gemini_backoff_multiplier = min(GEMINI_BACKOFF_MAX_MULT, gemini_backoff_multiplier * GEMINI_BACKOFF_STEP)
        except Exception:
            pass
        return ""

def parse_json_output(text):
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        data = json.loads(cleaned)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return 0.0, 0.0, 0.0, None, None, 'none', False, text
        try:
            data = json.loads(m.group())
        except Exception:
            return 0.0, 0.0, 0.0, None, None, 'none', False, text
    speed = float(data.get("speed_m_s", 0.0))
    steering = float(data.get("steering_deg", 0.0))
    distance = float(data.get("distance_m", 0.0))
    heading = data.get("heading_deg")
    try:
        heading = float(heading) if heading is not None else None
    except Exception:
        heading = None
    suggested_node = data.get("suggested_node")
    try:
        suggested_node = int(suggested_node) if suggested_node is not None else None
    except Exception:
        suggested_node = None
    traffic_light = data.get('traffic_light', 'none')
    stop_on_traffic = bool(data.get('stop_on_traffic_light', False))
    return speed, steering, distance, heading, suggested_node, traffic_light, stop_on_traffic, text

def to_pwm(speed, angle):
    # Map speed (m/s) into PWM percentage more aggressively for typical rover speeds
    # Assume reasonable driving speeds are in the 0-3 m/s range. Map that to 0-100%.
    base = (speed / 3.0) * 100.0
    # Keep steering influence moderate
    diff = (angle / 35.0) * 30.0
    left = np.clip(base + diff, -100, 100)
    right = np.clip(base - diff, -100, 100)
    return float(left), float(right)

def publish_cmd(client, l, r, d, duration_s):
    payload = json.dumps({
        "timestamp": int(time.time()*1000),
        "speed_left": l,
        "speed_right": r,
        "distance_m": d,
        "duration_s": duration_s
    })
    client.publish(TOPIC_CMD, payload, qos=1)
    # remember last published command and time so we can republish during Gemini backoff
    try:
        global last_published_cmd, last_published_time
        last_published_cmd = payload
        last_published_time = time.time()
    except Exception:
        pass
    print(f"📡 CMD → L={l:.1f}% R={r:.1f}% Dist={d:.2f}m Dur={duration_s:.2f}s")

# =========================
# MQTT
# =========================
def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected rc=", rc)
    client.subscribe(TOPIC_VIDEO)
    client.subscribe(TOPIC_OBS)
    # subscribe to planner and telemetry so Gemini can be constrained to the active waypoint
    try:
        client.subscribe(TOPIC_WAYPOINTS)
        client.subscribe(TOPIC_TELEMETRY)
    except Exception:
        pass

def on_message(client, userdata, msg):
    global prev_frame, frame_count, last_sent, rover_has_obstacle
    global waypoints, telemetry
    global gemini_backoff_until, last_published_cmd, last_published_time
    try:
        if msg.topic == TOPIC_OBS:
            try:
                d = json.loads(msg.payload.decode())
                rover_has_obstacle = bool(d.get("obstacle", False))
            except Exception:
                # tolerate simple boolean payloads
                try:
                    rover_has_obstacle = msg.payload.decode().strip().lower() in ("1", "true", "yes")
                except Exception:
                    rover_has_obstacle = False
            if rover_has_obstacle:
                print("⛔ Rover reports obstacle — will NOT query Gemini")
            return

        # waypoint update
        if msg.topic == TOPIC_WAYPOINTS:
            try:
                data = json.loads(msg.payload.decode())
                if 'waypoints' in data:
                    waypoints = data['waypoints']
                    print(f"[Gemini] Stored {len(waypoints)} planner waypoints")
            except Exception:
                pass
            return

        # telemetry update
        if msg.topic == TOPIC_TELEMETRY:
            try:
                telemetry = json.loads(msg.payload.decode())
            except Exception:
                telemetry = {}
            return

        # video frame
        frame = decode_image(msg.payload)
        cv2.imshow("Rover Feed", frame)

        # If rover already reports obstacle, skip everything
        if rover_has_obstacle:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client.disconnect(); cv2.destroyAllWindows()
            return

        # throttle model invocations
        if time.time() - last_sent < MIN_INTERVAL:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client.disconnect(); cv2.destroyAllWindows()
            return

        # If we're in a Gemini backoff window due to quota/rate-limit, skip model calls
        # and republish last known command as a keepalive so the rover can continue.
        try:
            if time.time() < gemini_backoff_until:
                if last_published_cmd and (time.time() - last_published_time) >= KEEPALIVE_REPUB_INTERVAL:
                    try:
                        client.publish(TOPIC_CMD, last_published_cmd, qos=1)
                        last_published_time = time.time()
                        print("[Gemini] republished last cmd as keepalive during backoff")
                    except Exception as e:
                        print("[Gemini] keepalive republish failed:", e)
                return
        except Exception:
            pass

        # If planner/telemetry context exists, always process frames so Gemini can act
        # on the active waypoint even when motion is subtle. Otherwise use motion
        # detection to reduce unnecessary model calls.
        process_frame = False
        try:
            if telemetry and waypoints:
                process_frame = True
            else:
                process_frame = detect_motion(frame, prev_frame, threshold=600)
        except Exception:
            process_frame = detect_motion(frame, prev_frame, threshold=600)

        if process_frame:
            frame_count += 1
            prev_frame = frame.copy()
            path = os.path.join(SAVE_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(path, frame)
            print(f"\n⚡ Motion detected → {path}")

            start = time.time()
            raw = get_prediction_from_gemini(frame)
            s,a,distance, heading, suggested_node, traffic_light, stop_on_traffic, raw_text = parse_json_output(raw)
            latency = time.time() - start
            print(f"--- Gemini --- s={s:.2f}, a={a:.1f}, d={distance:.2f}, heading={heading}, suggested_node={suggested_node}, traffic={traffic_light}, stop_on_traffic={stop_on_traffic}, latency={latency:.2f}s\n{raw_text}\n")

            # compute duration: distance / speed (if speed > 0) but cap to MAX_CMD_DURATION
            duration_s = 0.0
            if s > 0.05 and distance > 0.01:
                # speed s is m/s from the model; protect division by zero and unrealistic numbers
                safe_speed = max(s, 0.05)
                duration_s = distance / safe_speed
                # clip duration to a short value so rover can be responsive
                duration_s = float(min(duration_s, MAX_CMD_DURATION))
            else:
                duration_s = 0.0

            # If traffic light forces a stop, publish stop command
            if stop_on_traffic and traffic_light == 'red':
                publish_cmd(client, 0.0, 0.0, 0.0, 0.0)
                last_sent = time.time()
            else:
                # If model wants to stop (s small) then publish stop
                if s <= 0.05:
                    publish_cmd(client, 0.0, 0.0, 0.0, 0.0)
                    last_sent = time.time()
                else:
                    # convert to PWM % and send duration (ensure a minimum duration so rover safety doesn't kick in)
                    l, r = to_pwm(s, a)
                    # require a slightly longer minimum duration so rover receives and executes
                    # a continuous move (helps with intermittent network delays)
                    safe_dur = max(duration_s, 2.0)
                    safe_dur = float(min(safe_dur, MAX_CMD_DURATION))
                    publish_cmd(client, l, r, distance, safe_dur)
                    last_sent = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            client.disconnect(); cv2.destroyAllWindows()

    except Exception as e:
        print("❌ Frame error:", e)

# =========================
# Main
# =========================
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()
