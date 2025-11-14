# ==========================================================
# backend_server/server.py — OFFLINE GRAPH ROUTING + MQTT ROUTE PUSH
# ==========================================================
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from backend_server.mqtt_bridge import start_mqtt, send_command, send_route
from backend_server.rover_location_provider import (
    get_current_rover,
    update_rover_location
)
from backend_server.path_planner_offline import plan_from_coords


# ======================================================
# ENV
# ======================================================
load_dotenv()
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))

app = Flask(__name__)
CORS(app)

state = {
    "current_route": [],
    "dest": None,
    "last_route_ts": None,
    "gemini_logs": []
}

# ======================================================
# OFFLINE ROUTING: POST /set_destination_offline
# ======================================================
@app.route("/set_destination_offline", methods=["POST"])
def set_destination_offline():
    """
    body: { "dest_node": <int> }
    Computes offline Dijkstra route (cords.txt, list.txt).
    Publishes the result to MQTT → rover/route.
    """
    data = request.get_json(force=True)
    dest_node = data.get("dest_node")

    if dest_node is None:
        return jsonify({"error": "dest_node missing"}), 400

    cur = get_current_rover()
    if not cur:
        return jsonify({"error": "Rover location unknown"}), 400

    try:
        waypoints, meta = plan_from_coords(
            cur["lat"], cur["lon"], int(dest_node)
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    state["current_route"] = waypoints
    state["dest"] = {"node": int(dest_node)}
    state["last_route_ts"] = time.time()

    try:
        send_route(waypoints)
    except Exception as e:
        print("⚠️ MQTT route publish failed:", e)

    return jsonify({
        "waypoints_count": len(waypoints),
        "summary": meta,
        "first_waypoints": waypoints[:10]
    })

# ======================================================
# GET /get_route
# ======================================================
@app.route("/get_route", methods=["GET"])
def get_route():
    return jsonify({
        "waypoints": state["current_route"],
        "dest": state["dest"],
        "last_route_ts": state["last_route_ts"]
    })

# ======================================================
# Rover Location (HTTP fallback, optional)
# ======================================================
@app.route("/report_location", methods=["POST"])
def report_location():
    """
    Optional fallback.
    Usually rover sends GPS via MQTT.
    """
    data = request.get_json(force=True)
    lat = data.get("lat")
    lon = data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "missing lat/lon"}), 400

    update_rover_location(lat, lon)
    return jsonify({"ok": True})


@app.route("/get_rover_location", methods=["GET"])
def get_rover_location():
    return jsonify(get_current_rover() or {})

# ======================================================
# Gemini Logs
# ======================================================
@app.route("/log_gemini", methods=["POST"])
def log_gemini():
    out = request.json.get("out", "")
    state["gemini_logs"].append({"ts": time.time(), "out": out})
    state["gemini_logs"] = state["gemini_logs"][-200:]
    return jsonify({"ok": True})


@app.route("/get_gemini_logs", methods=["GET"])
def get_gemini_logs():
    return jsonify(state["gemini_logs"])

# ======================================================
# SEND DIRECT DRIVE COMMAND
# ======================================================
@app.route("/drive_command", methods=["POST"])
def drive_command():
    data = request.get_json(force=True)
    speed = data.get("speed", 0)
    steering = data.get("steering", 0)
    send_command(speed, steering)
    return jsonify({"ok": True})

# ======================================================
# START SERVER
# ======================================================
if __name__ == "__main__":
    print(f"🚀 Starting Rover Server on {SERVER_HOST}:{SERVER_PORT}")
    start_mqtt()  # this is required so MQTT updates rover location
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
