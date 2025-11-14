# rover_client/gemini_cache.py
import time
import hashlib
import json
import os
from collections import deque
from rover_client.gemini import _generate, log_to_server

CACHE_FILE = "/tmp/rover_gemini_cache.json"
RATE_LIMIT_PER_MIN = 6

try:
    with open(CACHE_FILE,'r') as f:
        CACHE = json.load(f)
except:
    CACHE = {}

_calls = deque()

def allowed():
    now = time.time()
    window_start = now - 60
    while _calls and _calls[0] < window_start:
        _calls.popleft()
    return len(_calls) < RATE_LIMIT_PER_MIN

def record():
    _calls.append(time.time())

def cached_decide(key_payload, prompt):
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
    if key in CACHE:
        return CACHE[key]
    if not allowed():
        return {"choice": "wait", "note": "rate limit reached"}
    try:
        record()
        out = _generate(prompt)
        CACHE[key] = out
        with open(CACHE_FILE,'w') as f:
            json.dump(CACHE, f)
        log_to_server(f"[rover cached] {str(out)[:200]}")
        return out
    except Exception as e:
        return {"error": str(e)}
