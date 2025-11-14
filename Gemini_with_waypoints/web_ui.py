from flask import Flask, Response, render_template_string, jsonify
import paho.mqtt.client as mqtt
import base64
import threading
import time

BROKER = "10.208.218.104"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_STATUS = "rover/status"
TOPIC_CONTROL = "rover/control"
TOPIC_TELEMETRY = "rover/telemetry"

app = Flask(__name__)
latest_frame = None
latest_status = None

HTML = """
<!doctype html>
<html>
    <head>
        <title>Rover Live Feed & Controls</title>
    </head>
    <body>
        <h1>Rover Live Camera</h1>
        <img id="stream" src="/video_feed" style="max-width:90%;height:auto;" />
        <h3>Controls</h3>
        <button onclick="control('start')">Start Auto</button>
        <button onclick="control('stop')">Stop</button>
        <button onclick="control('manual')">Manual</button>
        <h3>Telemetry</h3>
        <div>Min LIDAR Distance: <span id="min_dist">N/A</span> mm</div>
        <div>Current Waypoint Index: <span id="wp_idx">N/A</span></div>
        <pre id="status">{{ status }}</pre>

        <script>
            async function control(cmd){
                await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({cmd:cmd})});
            }

            // poll telemetry
            setInterval(async ()=>{
                try{
                    let r = await fetch('/telemetry');
                    let j = await r.json();
                    document.getElementById('min_dist').innerText = j.min_distance_mm || 'N/A';
                    document.getElementById('wp_idx').innerText = j.waypoint_index ?? 'N/A';
                }catch(e){}
            }, 500);
        </script>
    </body>
</html>
"""


def on_connect(client, userdata, flags, rc):
    print("[MQTT] Connected to broker", rc)
    client.subscribe(TOPIC_VIDEO)
    client.subscribe(TOPIC_STATUS)
    client.subscribe(TOPIC_TELEMETRY)


def on_message(client, userdata, msg):
    global latest_frame, latest_status
    try:
        if msg.topic == TOPIC_VIDEO:
            # payload is base64-encoded jpeg bytes
            try:
                # Some publishers send raw base64 bytes, some send str
                payload = msg.payload
                # If the payload is already raw image bytes, try to detect JPEG header
                if payload.startswith(b'\xff\xd8'):
                    latest_frame = payload
                else:
                    # otherwise assume base64 encoded
                    latest_frame = base64.b64decode(payload)
            except Exception as e:
                print('[WEB_UI] decode frame failed:', e)
        elif msg.topic == TOPIC_STATUS:
            try:
                latest_status = msg.payload.decode()
            except Exception:
                latest_status = str(msg.payload)
        elif msg.topic == TOPIC_TELEMETRY:
            try:
                latest_telemetry = msg.payload.decode()
                # telemetry stored as JSON; we simply keep latest string in status too
                latest_status = latest_telemetry
            except Exception:
                latest_status = str(msg.payload)
    except Exception as e:
        print('[WEB_UI] message handler error', e)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
threading.Thread(target=client.loop_forever, daemon=True).start()


@app.route("/")
def index():
    return render_template_string(HTML, status=latest_status or "No status yet")


@app.route('/control', methods=['POST'])
def control():
    from flask import request
    try:
        data = request.get_json() or {}
        cmd = data.get('cmd')
        if cmd:
            client.publish(TOPIC_CONTROL, cmd, qos=1)
            return ('', 204)
    except Exception as e:
        print('Control error', e)
    return ('', 400)


@app.route('/telemetry')
def telemetry():
    # The UI polls this endpoint; we mirror latest_status if it contains telemetry JSON
    import json as _json
    try:
        if latest_status:
            try:
                j = _json.loads(latest_status)
                return _json.dumps(j), 200, {'Content-Type':'application/json'}
            except Exception:
                return _json.dumps({'status': latest_status}), 200, {'Content-Type':'application/json'}
    except Exception:
        pass
    return _json.dumps({}), 200, {'Content-Type':'application/json'}


def gen_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        try:
            frame = latest_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            break
        except Exception as e:
            print('Frame gen error', e)
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(status=latest_status or "No status")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
