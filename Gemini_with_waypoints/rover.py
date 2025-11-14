#!/usr/bin/env python3
"""
Unified rover process:
- Captures camera frames
- Runs object detection (YOLO if available, else MOG2 motion)
- Publishes frames to MQTT topic `video/stream` for `gemini.py`
- Publishes obstacle and status messages
- Subscribes to `rover/cmd` to receive motor commands and executes them (serial or simulated)

This file merges and extends functionality from the original `r1.py` and ties into the navigation pipeline.
"""
import os
import time
import json
import base64
import threading
import sys

import cv2
import numpy as np
import math

import paho.mqtt.client as mqtt

# Attempt to import ultralytics YOLO model (optional)
USE_YOLO = False
try:
    from ultralytics import YOLO
    USE_YOLO = True
except Exception:
    YOLO = None

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
    except Exception as e:
        print('[Config] Failed to read config.json:', e)
        cfg = {}
else:
    print('[Config] config.json not found; falling back to environment variables')
    cfg = {}

# Configuration values (prefer config.json then env vars)
BROKER = cfg.get('mqtt', {}).get('broker', os.getenv('MQTT_BROKER', 'YOUR_BROKERIP'))
PORT = int(cfg.get('mqtt', {}).get('port', os.getenv('MQTT_PORT', '1883')))
TOPIC_VIDEO = 'video/stream'
TOPIC_CMD = 'rover/cmd'
TOPIC_STATUS = 'rover/status'
TOPIC_OBS = 'rover/obstacle'
TOPIC_WAYPOINTS = 'rover/waypoints'
TOPIC_TELEMETRY = 'rover/telemetry'

# Serial / motor controller
DDSM_PORT = cfg.get('ddsm_port', os.getenv('DDSM_PORT', '/dev/ttyACM1'))
SERIAL_BAUDRATE = int(cfg.get('serial_baudrate', os.getenv('SERIAL_BAUDRATE', '115200')))
SIMULATE = bool(cfg.get('allow_simulation', False))

# Camera
CAM_WIDTH, CAM_HEIGHT = 640, 480
# seconds between frames sent to model (make faster for lower-latency)
FRAME_INTERVAL = float(cfg.get('frame_interval_s', 0.5))  # send frames faster to reduce latency (default 0.5s)

# Path / obstacle tuning (tune on real rover)
# fraction of image width considered the central path corridor (centered)
PATH_WIDTH_FRAC = float(cfg.get('path_width_frac', 0.4))
# fraction of image height; bbox bottom must be below this to be considered 'on-path' (nearer to rover)
PATH_BOTTOM_THRESH = float(cfg.get('path_bottom_frac', 0.6))
# empirical constant for distance estimation: distance_m ≈ DIST_EST_K / bbox_height_px
# This must be calibrated per camera+mounting; default is conservative and likely needs tuning
DIST_EST_K = float(cfg.get('dist_est_k', 2000.0))
# assumed max travel speed (m/s) when issuing timed forward movements
MAX_SPEED_M_S = float(cfg.get('max_speed_m_s', 0.8))
# minimum distance to trigger the "move N meters then stop" behavior
MIN_MOVE_METERS = float(cfg.get('min_move_m', 2.0))

# YOLO model path (optional)
YOLO_PATH = cfg.get('yolo_path', os.getenv('YOLO_PATH', 'yolov8n.pt'))

# Globals
motor_left = 0.0
motor_right = 0.0
target_distance = 0.0
executing = False
last_cmd_time = time.time()
stop_timer = None
obstacle_detected = False
# last received telemetry (populated from MQTT) and lock
last_telemetry = None
last_telemetry_time = 0.0
telemetry_lock = threading.Lock()
# timer used for autonomous move-N-meters actions (separate from command stop_timer)
move_stop_timer = None
move_in_progress = False

# Load YOLO if available and model file exists
model = None
if USE_YOLO and os.path.exists(YOLO_PATH):
    try:
        model = YOLO(YOLO_PATH)
        print(f"[YOLO] Loaded model {YOLO_PATH}")
    except Exception as e:
        print("[YOLO] Failed to load model:", e)
        model = None
else:
    print('[YOLO] Not available, using MOG2 motion detection fallback')
    mog = cv2.createBackgroundSubtractorMOG2(300, 16, False)

# Serial handle (open when needed)
serial_dev = None
if not SIMULATE:
    try:
        import serial as _serial
    except Exception as e:
        print('[Serial] pyserial not installed or unavailable:', e)
        print('Hardware mode requested in config but pyserial missing. Install pyserial or set allow_simulation=true in config.json')
        sys.exit(1)
    try:
        serial_dev = _serial.Serial(DDSM_PORT, SERIAL_BAUDRATE, timeout=1)
        serial_dev.setRTS(False)
        serial_dev.setDTR(False)
        print('[Serial] Motor controller connected on', DDSM_PORT)
    except Exception as e:
        print('[Serial] cannot open serial port', DDSM_PORT, ':', e)
        print('Hardware mode requested in config but motor serial device not available. Check port and permissions.')
        sys.exit(1)

# Helper to build motor command JSON
def build_motor_command(mid, val):
    return {"T": 10010, "id": mid, "cmd": int(val), "act": 3}

# Motor control: writes JSON lines to serial or prints (simulate)
def motor_control(left, right):
    global serial_dev
    global last_cmd_time, last_motor_left, last_motor_right, executing
    cmd_r = build_motor_command(2, -int(right))
    cmd_l = build_motor_command(1, int(left))
    if serial_dev is not None:
        try:
            serial_dev.write((json.dumps(cmd_r) + '\n').encode())
            time.sleep(0.01)
            serial_dev.write((json.dumps(cmd_l) + '\n').encode())
        except Exception as e:
            print('[Motor] Serial write failed:', e)
    else:
        print(f'[SIM MOTOR] L={cmd_l["cmd"]} R={-cmd_r["cmd"]}')
    # update last-known motor outputs and keepalive timestamp
    last_motor_left = int(left)
    last_motor_right = int(right)
    last_cmd_time = time.time()
    executing = (left != 0 or right != 0)

# MQTT callbacks
client = mqtt.Client()

# Stored waypoints (list of dicts: {'lat':..., 'lon':..., 'alt':...})
waypoints = []
nav_index = 0
NAV_TOLERANCE_METERS = float(cfg.get('nav_tolerance_m', 1.0))
# Default higher base speed to move faster on open ground
BASE_SPEED_PCT = float(cfg.get('base_speed_pct', 70.0))
STEER_KP = float(cfg.get('steer_kp', 0.9))
# safety timeout: if no nav/command updates within this many seconds, stop
SAFETY_TIMEOUT = float(cfg.get('safety_timeout_s', 20.0))

# Navigation loop rate (Hz) - how often rover evaluates nav_step when telemetry available
NAV_RATE = float(cfg.get('nav_rate_hz', 5.0))

# last motor outputs (for keepalive resends)
last_motor_left = 0
last_motor_right = 0

# Obstacle detection area fraction to consider as 'large'
OBSTACLE_AREA_FRAC = float(cfg.get('obstacle_area_frac', 0.06))

# Obstacle debounce counters
obstacle_counter = 0
OBSTACLE_DEBOUNCE = int(cfg.get('obstacle_debounce', 3))


def on_connect(c, userdata, flags, rc):
    print('[MQTT] Connected', rc)
    # Subscribe to both motor commands and planner waypoints
    try:
        c.subscribe(TOPIC_CMD)
        c.subscribe(TOPIC_WAYPOINTS)
        c.subscribe(TOPIC_TELEMETRY)
    except Exception as e:
        print('[MQTT] subscribe failed', e)


def handle_command(msg_payload):
    global motor_left, motor_right, target_distance, executing, last_cmd_time, last_telemetry
    global stop_timer
    try:
        data = json.loads(msg_payload.decode()) if isinstance(msg_payload, (bytes, bytearray)) else json.loads(msg_payload)
        cmd_left = float(data.get('speed_left', 0))
        cmd_right = float(data.get('speed_right', 0))
        target_distance = float(data.get('distance_m', 0))
        duration = float(data.get('duration_s', 0)) if data.get('duration_s') is not None else 0

        # ensure a minimum duration to avoid safety timeouts
        duration = max(duration, 1.0)

        # If we have recent telemetry and waypoints, validate the command is toward the active waypoint
        allow_cmd = True
        try:
            with telemetry_lock:
                lt = last_telemetry
            if lt and waypoints and nav_index < len(waypoints):
                lat = float(lt.get('lat'))
                lon = float(lt.get('lon'))
                heading = float(lt.get('heading', 0.0))
                wp = waypoints[nav_index]
                wplat = float(wp.get('lat'))
                wplon = float(wp.get('lon'))
                target_brg = bearing_between(lat, lon, wplat, wplon)

                # estimate steering degrees from PWM percent mapping used by Gemini
                # reverse of to_pwm: angle ≈ (left - right) * 0.35
                steering_deg_est = (cmd_left - cmd_right) * 0.35
                new_heading = (heading + steering_deg_est) % 360.0
                # signed diffs
                def signed_diff(a, b):
                    return ((a - b + 180 + 360) % 360) - 180

                diff_before = signed_diff(target_brg, heading)
                diff_after = signed_diff(target_brg, new_heading)

                # Accept command if it reduces angular error or keeps it within a small margin
                if abs(diff_after) > abs(diff_before) + 10.0:
                    allow_cmd = False
                    print(f"[CMD-REJECT] steering would increase path error: before={diff_before:.1f}° after={diff_after:.1f}°")
        except Exception:
            allow_cmd = True

        if not allow_cmd:
            # reject the command; optionally publish a stop/keepalive
            motor_control(0, 0)
            try:
                client.publish(TOPIC_STATUS, json.dumps({'event': 'cmd_rejected', 'reason': 'off_path'}), qos=0)
            except Exception:
                pass
            return

        # accept command
        motor_left = cmd_left
        motor_right = cmd_right
        executing = (motor_left != 0 or motor_right != 0)
        last_cmd_time = time.time()
        print(f"[CMD] L={motor_left} R={motor_right} dist={target_distance} dur={duration}")

        # Use a non-blocking timer to stop motors after `duration` seconds so the MQTT
        # callback thread is not blocked by time.sleep (which previously caused slow
        # command handling and delayed reception of subsequent commands).
        if executing and duration and duration > 0:
            try:
                if stop_timer is not None:
                    stop_timer.cancel()
            except Exception:
                pass
            motor_control(motor_left, motor_right)
            t = threading.Timer(min(duration, 60.0), lambda: motor_control(0, 0))
            t.daemon = True
            t.start()
            stop_timer = t
        elif executing:
            motor_control(motor_left, motor_right)
        else:
            motor_control(0, 0)
    except Exception as e:
        print('[MQTT] command parse error', e)


def handle_waypoints(msg_payload):
    global waypoints
    try:
        # payload can be bytes (JSON) or already decoded
        data = json.loads(msg_payload.decode()) if isinstance(msg_payload, (bytes, bytearray)) else json.loads(msg_payload)
        if 'waypoints' in data and isinstance(data['waypoints'], list):
            waypoints = data['waypoints']
            print(f"[WAYPOINTS] Received {len(waypoints)} waypoints")
            # Acknowledge receipt
            try:
                client.publish(TOPIC_STATUS, json.dumps({'event': 'waypoints_received', 'count': len(waypoints)}), qos=1)
            except Exception:
                pass
        else:
            print('[WAYPOINTS] invalid payload format')
    except Exception as e:
        print('[WAYPOINTS] parse error', e)


def haversine_m(lat1, lon1, lat2, lon2):
    # returns distance in meters between two lat/lon points
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def bearing_between(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)
    y = math.sin(lam2-lam1) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(lam2-lam1)
    br = math.degrees(math.atan2(y, x))
    return (br + 360.0) % 360.0


def clamp(v, a, b):
    return max(a, min(b, v))


def nav_step(lat, lon, heading):
    """Compute motor outputs to drive toward current waypoint using a simple P steering controller."""
    global nav_index, waypoints, executing, last_cmd_time
    if not waypoints:
        return
    if nav_index >= len(waypoints):
        # finished
        motor_control(0, 0)
        return

    wp = waypoints[nav_index]
    try:
        wplat = float(wp.get('lat'))
        wplon = float(wp.get('lon'))
    except Exception:
        print('[NAV] invalid waypoint format, skipping')
        nav_index += 1
        return

    dist = haversine_m(lat, lon, wplat, wplon)
    target_brg = bearing_between(lat, lon, wplat, wplon)
    # compute smallest signed angle difference
    diff = ((target_brg - heading + 180 + 360) % 360) - 180

    # If within tolerance, advance to next waypoint
    if dist <= NAV_TOLERANCE_METERS:
        print(f"[NAV] reached waypoint {nav_index} (dist {dist:.2f} m)")
        nav_index += 1
        motor_control(0, 0)
        return

    # compute steering adjustment: map angle to differential percent
    steer_adj = (diff / 35.0) * 50.0 * STEER_KP
    left = clamp(BASE_SPEED_PCT + steer_adj, -100, 100)
    right = clamp(BASE_SPEED_PCT - steer_adj, -100, 100)

    # apply obstacle safety: if obstacle_detected, stop
    if obstacle_detected:
        motor_control(0, 0)
        return

    # send motor outputs and update keepalive
    motor_control(int(left), int(right))
    executing = True
    last_cmd_time = time.time()
    # also publish status for debugging
    try:
        client.publish(TOPIC_STATUS, json.dumps({'event': 'nav_cmd', 'left': left, 'right': right, 'dist_m': dist, 'wp_idx': nav_index}), qos=0)
    except Exception:
        pass


def on_message(c, userdata, msg):
    # Route messages by topic
    global last_telemetry, last_telemetry_time
    if msg.topic == TOPIC_CMD:
        handle_command(msg.payload)
    elif msg.topic == TOPIC_TELEMETRY:
        # telemetry should be JSON with lat, lon, heading
        try:
            data = json.loads(msg.payload.decode())
            with telemetry_lock:
                last_telemetry = data
                last_telemetry_time = time.time()
        except Exception as e:
            print('[TELEM] parse error', e)
    elif msg.topic == TOPIC_WAYPOINTS:
        handle_waypoints(msg.payload)
    else:
        print('[MQTT] message on unknown topic', msg.topic)


client.on_connect = on_connect
client.on_message = on_message

# Obstacle detection helper

def detect_obstacle(frame, lidar_distance=None):
    # Return structured detection info: {'any': bool, 'on_path': bool, 'min_distance': float|None, 'details': [...]}
    info = {'any': False, 'on_path': False, 'min_distance': None, 'details': []}
    h, w = frame.shape[:2]

    # If a close lidar reading exists, convert that into immediate detection
    if lidar_distance is not None:
        info['any'] = True
        info['min_distance'] = float(lidar_distance)
        info['on_path'] = lidar_distance < 1.5  # assume close lidar reading is relevant
        info['details'].append({'source': 'lidar', 'distance_m': float(lidar_distance)})
        return info

    # YOLO-based detection (preferred)
    if model is not None:
        try:
            res = model(frame, verbose=False)
            min_dist = None
            for det in getattr(res[0], 'boxes', []) or []:
                try:
                    conf = float(det.conf[0]) if hasattr(det, 'conf') else float(det.conf)
                except Exception:
                    conf = 0.0
                if conf < 0.35:
                    continue
                # box coordinates
                try:
                    xy = det.xyxy[0]
                    x1, y1, x2, y2 = map(int, xy)
                except Exception:
                    # fallback extraction
                    vals = list(map(int, det.xyxy))
                    if len(vals) >= 4:
                        x1, y1, x2, y2 = vals[:4]
                    else:
                        continue

                bw = x2 - x1
                bh = max(1, y2 - y1)
                cx = (x1 + x2) / 2.0
                bottom = y2

                # determine if this bbox is within the forward path corridor
                corridor_left = w * (0.5 - PATH_WIDTH_FRAC / 2.0)
                corridor_right = w * (0.5 + PATH_WIDTH_FRAC / 2.0)
                is_center = (cx >= corridor_left and cx <= corridor_right)
                is_near_bottom = (bottom >= h * PATH_BOTTOM_THRESH)
                on_path = is_center and is_near_bottom

                # estimate distance (empirical): larger bbox height => closer
                dist_m = None
                try:
                    dist_m = float(DIST_EST_K) / float(bh)
                except Exception:
                    dist_m = None

                info['details'].append({'source': 'yolo', 'conf': conf, 'bbox': [x1, y1, x2, y2], 'on_path': on_path, 'est_m': dist_m})
                info['any'] = True
                if on_path:
                    info['on_path'] = True
                if dist_m is not None:
                    if min_dist is None or dist_m < min_dist:
                        min_dist = dist_m

            if min_dist is not None:
                info['min_distance'] = float(min_dist)
            return info
        except Exception as e:
            print('[YOLO] detection error', e)

    # MOG2 motion detection fallback (produce structured info similar to YOLO)
    try:
        mask = mog.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = h * w
        min_dist = None
        for c in contours:
            if cv2.contourArea(c) < area * OBSTACLE_AREA_FRAC:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            cx = x + cw/2.0
            bottom = y + ch
            corridor_left = w * (0.5 - PATH_WIDTH_FRAC / 2.0)
            corridor_right = w * (0.5 + PATH_WIDTH_FRAC / 2.0)
            is_center = (cx >= corridor_left and cx <= corridor_right)
            is_near_bottom = (bottom >= h * PATH_BOTTOM_THRESH)
            on_path = is_center and is_near_bottom
            dist_m = None
            try:
                dist_m = float(DIST_EST_K) / float(ch)
            except Exception:
                dist_m = None
            info['details'].append({'source': 'mog2', 'area': cv2.contourArea(c), 'bbox': [int(x), int(y), int(x+cw), int(y+ch)], 'on_path': on_path, 'est_m': dist_m})
            info['any'] = True
            if on_path:
                info['on_path'] = True
            if dist_m is not None:
                if min_dist is None or dist_m < min_dist:
                    min_dist = dist_m
        if min_dist is not None:
            info['min_distance'] = float(min_dist)
        return info
    except Exception as e:
        print('[MOG2] detection error', e)
        return info


def publish_status(status):
    try:
        payload = json.dumps({"timestamp": int(time.time()*1000), "status": status})
        client.publish(TOPIC_STATUS, payload, qos=1)
    except Exception as e:
        print('[MQTT] publish_status failed', e)


def run_camera_loop():
    # we modify these module-level variables from this thread
    global obstacle_detected, last_cmd_time, obstacle_counter, executing, move_stop_timer, move_in_progress
    # Try camera backends in order to tolerate different platforms / drivers
    cam_opened = False
    cap = None
    # 1) NVIDIA Jetson pipeline (nvarguscamerasrc) - only available on Jetson
    try:
        gst = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=%d,height=%d,framerate=30/1 ! "
            "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! videoconvert ! appsink" % (CAM_WIDTH, CAM_HEIGHT)
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print('[Camera] opened nvarguscamerasrc (GStreamer)')
            cam_opened = True
    except Exception:
        cap = None

    # 2) Try index 0 (V4L2 / default webcam)
    if not cam_opened:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print('[Camera] opened device index 0')
                cam_opened = True
        except Exception:
            cap = None

    # 3) Try common device paths
    if not cam_opened:
        for dev in ('/dev/video0', '/dev/video1'):
            try:
                cap = cv2.VideoCapture(dev)
                if cap.isOpened():
                    print(f'[Camera] opened device {dev}')
                    cam_opened = True
                    break
            except Exception:
                cap = None

    if not cam_opened or cap is None or not cap.isOpened():
        print('[Camera] failed to open any camera source. If you are on a Jetson, ensure the camera is connected and nvargus is available. If on Linux, check /dev/video* and permissions.')
        return

    last_send = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # detect obstacle with debounce (use structured detection)
            lidar_distance = None
            det = detect_obstacle(frame, lidar_distance)

            # only debounce when detection is on the forward path
            if det.get('on_path'):
                obstacle_counter = min(obstacle_counter + 1, OBSTACLE_DEBOUNCE)
            else:
                obstacle_counter = max(obstacle_counter - 1, 0)

            # publish lightweight obstacle info for monitoring (including on_path flag)
            try:
                client.publish(TOPIC_OBS, json.dumps({'obstacle_any': bool(det.get('any')), 'on_path': bool(det.get('on_path')), 'min_distance_m': det.get('min_distance')}), qos=0)
            except Exception:
                pass

            # When debounce threshold reached, act depending on estimated range
            if obstacle_counter >= OBSTACLE_DEBOUNCE and not obstacle_detected:
                obstacle_detected = True
                print('[OBSTACLE] on-path detected — evaluating response')
                publish_status('OBSTACLE_ON_PATH')

                # if distance known and reasonably far, move forward MIN_MOVE_METERS then stop
                md = det.get('min_distance')
                if md is not None and md >= MIN_MOVE_METERS:
                    # schedule a non-blocking forward movement for MIN_MOVE_METERS
                    if not move_in_progress:
                        move_in_progress = True
                        move_duration = max(0.5, MIN_MOVE_METERS / max(0.01, MAX_SPEED_M_S))
                        print(f'[OBSTACLE] estimated {md:.2f} m away — moving {MIN_MOVE_METERS} m (≈{move_duration:.2f}s) then stop')
                        try:
                            # start forward movement at base speed
                            motor_control(int(BASE_SPEED_PCT), int(BASE_SPEED_PCT))
                        except Exception:
                            pass

                        def _end_move():
                            global move_in_progress
                            try:
                                motor_control(0, 0)
                            except Exception:
                                pass
                            move_in_progress = False
                            publish_status('MOVE_FINISHED')

                        try:
                            if move_stop_timer is not None:
                                try:
                                    move_stop_timer.cancel()
                                except Exception:
                                    pass
                            tmove = threading.Timer(min(move_duration, 120.0), _end_move)
                            tmove.daemon = True
                            tmove.start()
                            move_stop_timer = tmove
                        except Exception:
                            move_in_progress = False
                else:
                    # immediate stop if object is closer than MIN_MOVE_METERS or distance unknown
                    print('[OBSTACLE] close or unknown distance — stopping motors')
                    try:
                        motor_control(0, 0)
                    except Exception:
                        pass
                    publish_status('STOPPED_OBSTACLE')

            elif obstacle_counter == 0 and obstacle_detected:
                obstacle_detected = False
                publish_status('PATH_CLEARED')

            # send periodic frames to Gemini (downscale to reduce latency)
            if time.time() - last_send >= FRAME_INTERVAL:
                # downscale for faster network/model processing (smaller width for low latency)
                target_w = 320
                small = cv2.resize(frame, (target_w, int(target_w * frame.shape[0] / frame.shape[1])))
                # slightly lower quality to reduce payload size and latency
                _, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                try:
                    client.publish(TOPIC_VIDEO, base64.b64encode(buf.tobytes()), qos=0)
                except Exception as e:
                    print('[MQTT] publish frame failed', e)
                last_send = time.time()

            # safety timeout for motor commands (configurable)
            if time.time() - last_cmd_time > SAFETY_TIMEOUT and executing:
                print('[SAFETY] command timeout — stopping motors')
                motor_control(0, 0)

            # show local preview optionally
            if os.getenv('SHOW_LOCAL', '0') == '1':
                cv2.imshow('Rover Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print('\n[Camera] stopped')
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def nav_loop_thread():
    """Background navigation loop: uses last telemetry at NAV_RATE to call nav_step.
    If an external command was recently applied (executing True and last_cmd_time recent),
    we let that command run and skip the nav_step to avoid command fights.
    """
    global last_telemetry, last_telemetry_time, executing
    interval = 1.0 / max(0.1, float(NAV_RATE))
    while True:
        try:
            now = time.time()
            # prefer external commands if they were sent very recently
            if executing and (now - last_cmd_time) < 1.0:
                # let external command run
                time.sleep(interval)
                continue

            with telemetry_lock:
                lt = last_telemetry
            if lt:
                try:
                    lat = float(lt.get('lat'))
                    lon = float(lt.get('lon'))
                    heading = float(lt.get('heading', 0.0))
                    nav_step(lat, lon, heading)
                except Exception:
                    pass
            else:
                # no telemetry available: if we have recent motor outputs and we are executing,
                # resend last motor command as a keepalive to prevent SAFETY timeout
                try:
                    if executing and (time.time() - last_cmd_time) > (interval * 0.9):
                        motor_control(last_motor_left, last_motor_right)
                except Exception:
                    pass

            time.sleep(interval)
        except Exception:
            time.sleep(interval)


def main():
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print('[MQTT] connect failed', e)
        return
    client.loop_start()

    # start camera and navigation threads
    cam_thread = threading.Thread(target=run_camera_loop, daemon=True)
    cam_thread.start()

    nav_thread = threading.Thread(target=nav_loop_thread, daemon=True)
    nav_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[ROVER] stopping')
    finally:
        try:
            client.loop_stop()
            if serial_dev is not None:
                serial_dev.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
