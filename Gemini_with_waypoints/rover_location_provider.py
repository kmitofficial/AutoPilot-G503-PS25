# backend_server/rover_location_provider.py
import time
from threading import Lock

_latest = {"lat": None, "lon": None, "ts": None}
_lock = Lock()

def update_rover_location(lat, lon):
    with _lock:
        _latest["lat"] = float(lat)
        _latest["lon"] = float(lon)
        _latest["ts"] = time.time()

def get_current_rover():
    with _lock:
        if _latest["lat"] is None:
            return None
        return {"lat": _latest["lat"], "lon": _latest["lon"], "ts": _latest["ts"]}
