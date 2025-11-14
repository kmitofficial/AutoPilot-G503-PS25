# import socketio
# import eventlet
# import numpy as np
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image
# import cv2
 
# sio = socketio.Server()
 
# app = Flask(__name__) #'__main__'
# speed_limit = 10
# def img_preprocess(img):
#     img = img[60:135,:,:]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = cv2.GaussianBlur(img,  (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     img = img/255
#     return img
 
 
# @sio.on('telemetry')
# def telemetry(sid, data):
#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.asarray(image)
#     image = img_preprocess(image)
#     image = np.array([image])
#     steering_angle = float(model.predict(image))
#     throttle = 1.0 - speed/speed_limit
#     print('{} {} {}'.format(steering_angle, throttle, speed))
#     send_control(steering_angle, throttle)
 
 
 
# @sio.on('connect')
# def connect(sid, environ):
#     print('Connected')
#     send_control(0, 0)
 
# def send_control(steering_angle, throttle):
#     sio.emit('steer', data = {
#         'steering_angle': steering_angle.__str__(),
#         'throttle': throttle.__str__()
#     })
 
 
# if __name__ == '__main__':
#     model = load_model('model/model.h5')
#     app = socketio.Middleware(sio, app)
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


# // Revised code with virtual traffic-light logic and waypoint navigation 
# import socketio
# import eventlet
# import numpy as np
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image
# import cv2
# import time
# import random

# # ----------------------------------------------------------
# #  SocketIO + Flask setup
# # ----------------------------------------------------------
# sio = socketio.Server()
# app = Flask(__name__)

# # ----------------------------------------------------------
# #  Parameters
# # ----------------------------------------------------------
# speed_limit = 10
# model = None

# # ----------------------------------------------------------
# #  Virtual traffic-light logic
# # ----------------------------------------------------------
# current_light = {"color": "green", "last_change": time.time()}
# def get_virtual_light():
#     now = time.time()
#     if now - current_light["last_change"] > 6:
#         current_light["color"] = random.choice(["red", "yellow", "green"])
#         current_light["last_change"] = now
#     return current_light["color"]

# # ----------------------------------------------------------
# #  Waypoints (fake route)
# # ----------------------------------------------------------
# waypoints = [(50, 150), (200, 150), (350, 200), (500, 200), (550, 150)]
# current_wp = 0
# car_pos = list(waypoints[0])
# reached = False

# #  Simple mini-map image
# map_img = np.zeros((300, 600, 3), np.uint8)

# # ----------------------------------------------------------
# #  Image preprocessing
# # ----------------------------------------------------------
# def img_preprocess(img):
#     img = img[60:135, :, :]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = cv2.GaussianBlur(img, (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     img = img / 255.0
#     return img

# # ----------------------------------------------------------
# #  Move car along waypoints
# # ----------------------------------------------------------
# def update_position():
#     global car_pos, current_wp, reached
#     if reached or current_wp >= len(waypoints):
#         return
#     target = np.array(waypoints[current_wp])
#     car = np.array(car_pos)
#     dir_vec = target - car
#     dist = np.linalg.norm(dir_vec)
#     if dist < 5:
#         current_wp += 1
#         if current_wp >= len(waypoints):
#             reached = True
#             print("✅ Destination reached!")
#             return
#     else:
#         step = 3 * dir_vec / (dist + 1e-5)
#         car_pos[0] += step[0]
#         car_pos[1] += step[1]

# # ----------------------------------------------------------
# #  Draw mini map
# # ----------------------------------------------------------
# def draw_map():
#     canvas = map_img.copy()
#     # Draw path
#     for p in waypoints:
#         cv2.circle(canvas, tuple(map(int, p)), 4, (80, 80, 80), -1)
#     # Draw destination
#     cv2.circle(canvas, tuple(map(int, waypoints[-1])), 8, (0, 0, 255), -1)
#     # Draw car
#     cv2.circle(canvas, tuple(map(int, car_pos)), 8, (0, 255, 0), -1)
#     cv2.imshow("Mini-Map: Route", canvas)
#     cv2.waitKey(1)

# # ----------------------------------------------------------
# #  Telemetry handler
# # ----------------------------------------------------------
# @sio.on('telemetry')
# def telemetry(sid, data):
#     global model, reached
#     if reached:
#         send_control(0, 0)
#         return

#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.array(image).copy()

#     color = get_virtual_light()
#     if color == "red":
#         cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
#     elif color == "yellow":
#         cv2.circle(image, (50, 50), 15, (0, 255, 255), -1)
#     else:
#         cv2.circle(image, (50, 50), 15, (0, 255, 0), -1)

#     proc_img = img_preprocess(image)
#     proc_img = np.array([proc_img])
#     steering_angle = float(model.predict(proc_img))

#     if color == "red":
#         throttle = 0.0
#     elif color == "yellow":
#         throttle = 0.2
#     else:
#         throttle = 1.0 - speed / speed_limit
#         throttle = max(min(throttle, 1.0), 0.0)

#     update_position()
#     draw_map()

#     print(f"{color.upper():7s} | WP={current_wp}/{len(waypoints)} | "
#           f"steer={steering_angle:+.3f} | throttle={throttle:.2f}")

#     send_control(steering_angle, throttle)

# # ----------------------------------------------------------
# #  Connect handler
# # ----------------------------------------------------------
# @sio.on('connect')
# def connect(sid, environ):
#     print('Connected to simulator')
#     send_control(0, 0)

# # ----------------------------------------------------------
# #  Send control
# # ----------------------------------------------------------
# def send_control(steering_angle, throttle):
#     sio.emit('steer', data={
#         'steering_angle': str(steering_angle),
#         'throttle': str(throttle)
#     })

# # ----------------------------------------------------------
# #  Main
# # ----------------------------------------------------------
# if __name__ == '__main__':
#     model = load_model('model/model.h5')
#     print("Model loaded successfully.")
#     app = socketio.Middleware(sio, app)
#     print("Server listening on port 4567 ...")
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


# import socketio
# import eventlet
# import numpy as np
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image
# import cv2
# import time
# import random
# import heapq

# # ----------------------------------------------------------
# #  SocketIO + Flask setup
# # ----------------------------------------------------------
# sio = socketio.Server()
# app = Flask(__name__)

# # ----------------------------------------------------------
# #  Parameters
# # ----------------------------------------------------------
# speed_limit = 10
# model = None

# # ----------------------------------------------------------
# #  Virtual traffic-light logic
# # ----------------------------------------------------------
# current_light = {"color": "green", "last_change": time.time()}

# def get_virtual_light():
#     now = time.time()
#     if now - current_light["last_change"] > 6:
#         current_light["color"] = random.choice(["red", "yellow", "green"])
#         current_light["last_change"] = now
#     return current_light["color"]

# # ----------------------------------------------------------
# #  Graph setup for route planning (Dijkstra)
# # ----------------------------------------------------------
# graph = {
#     'A': {'B': 5, 'C': 8},
#     'B': {'A': 5, 'D': 7},
#     'C': {'A': 8, 'D': 3, 'E': 4},
#     'D': {'B': 7, 'C': 3, 'E': 2},
#     'E': {'C': 4, 'D': 2}
# }

# positions = {
#     'A': (50, 150),
#     'B': (200, 150),
#     'C': (100, 220),
#     'D': (350, 200),
#     'E': (550, 150)
# }

# def dijkstra(graph, start, goal):
#     queue = [(0, start, [])]
#     seen = set()
#     while queue:
#         (cost, node, path) = heapq.heappop(queue)
#         if node in seen:
#             continue
#         path = path + [node]
#         seen.add(node)
#         if node == goal:
#             return (cost, path)
#         for n, weight in graph[node].items():
#             if n not in seen:
#                 heapq.heappush(queue, (cost + weight, n, path))
#     return (float("inf"), [])

# # ----------------------------------------------------------
# #  Choose source and destination
# # ----------------------------------------------------------
# source, destination = 'A', 'E'
# cost, route_nodes = dijkstra(graph, source, destination)
# print(f"Selected route: {' -> '.join(route_nodes)}  (cost={cost})")

# waypoints = [positions[n] for n in route_nodes]
# current_wp = 0
# car_pos = list(waypoints[0])
# reached = False

# # Mini-map canvas
# map_img = np.zeros((300, 600, 3), np.uint8)

# # ----------------------------------------------------------
# #  Image preprocessing
# # ----------------------------------------------------------
# def img_preprocess(img):
#     img = img[60:135, :, :]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = cv2.GaussianBlur(img, (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     img = img / 255.0
#     return img

# # ----------------------------------------------------------
# #  Move car along route
# # ----------------------------------------------------------
# def update_position():
#     global car_pos, current_wp, reached
#     if reached or current_wp >= len(waypoints):
#         return
#     target = np.array(waypoints[current_wp])
#     car = np.array(car_pos)
#     dir_vec = target - car
#     dist = np.linalg.norm(dir_vec)
#     if dist < 5:
#         current_wp += 1
#         if current_wp >= len(waypoints):
#             reached = True
#             print("✅ Destination reached!")
#             return
#     else:
#         step = 3 * dir_vec / (dist + 1e-5)
#         car_pos[0] += step[0]
#         car_pos[1] += step[1]

# # ----------------------------------------------------------
# #  Draw map with nodes and routes
# # ----------------------------------------------------------
# def draw_map():
#     canvas = map_img.copy()

#     # Draw all edges
#     for a, neighbors in graph.items():
#         for b in neighbors:
#             cv2.line(canvas, positions[a], positions[b], (80, 80, 80), 2)

#     # Draw nodes
#     for name, pos in positions.items():
#         color = (200, 200, 200)
#         if name in route_nodes:
#             color = (255, 255, 0)
#         cv2.circle(canvas, pos, 6, color, -1)
#         cv2.putText(canvas, name, (pos[0]-10, pos[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # Draw car + destination
#     cv2.circle(canvas, tuple(map(int, car_pos)), 8, (0, 255, 0), -1)
#     cv2.circle(canvas, positions[destination], 8, (0, 0, 255), -1)

#     cv2.imshow("Mini-Map: Route Planning", canvas)
#     cv2.waitKey(1)

# # ----------------------------------------------------------
# #  Telemetry handler
# # ----------------------------------------------------------
# @sio.on('telemetry')
# def telemetry(sid, data):
#     global model, reached
#     if reached:
#         send_control(0, 0)
#         return

#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.array(image).copy()

#     # Virtual traffic light color
#     color = get_virtual_light()
#     if color == "red":
#         cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
#     elif color == "yellow":
#         cv2.circle(image, (50, 50), 15, (0, 255, 255), -1)
#     else:
#         cv2.circle(image, (50, 50), 15, (0, 255, 0), -1)

#     # Model steering prediction
#     proc_img = img_preprocess(image)
#     proc_img = np.array([proc_img])
#     steering_angle = float(model.predict(proc_img))

#     # Throttle logic
#     if color == "red":
#         throttle = 0.0
#     elif color == "yellow":
#         throttle = 0.2
#     else:
#         throttle = 1.0 - speed / speed_limit
#         throttle = max(min(throttle, 1.0), 0.0)

#     # Move and draw
#     update_position()
#     draw_map()

#     print(f"{color.upper():7s} | WP={current_wp}/{len(waypoints)} | "
#           f"steer={steering_angle:+.3f} | throttle={throttle:.2f}")

#     send_control(steering_angle, throttle)

# # ----------------------------------------------------------
# #  On connect
# # ----------------------------------------------------------
# @sio.on('connect')
# def connect(sid, environ):
#     print('Connected to simulator')
#     send_control(0, 0)

# # ----------------------------------------------------------
# #  Send control data
# # ----------------------------------------------------------
# def send_control(steering_angle, throttle):
#     sio.emit('steer', data={
#         'steering_angle': str(steering_angle),
#         'throttle': str(throttle)
#     })

# # ----------------------------------------------------------
# #  Main
# # ----------------------------------------------------------
# if __name__ == '__main__':
#     model = load_model('model/model.h5')
#     print("Model loaded successfully.")
#     app = socketio.Middleware(sio, app)
#     print("Server listening on port 4567 ...")
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import time
import random
import heapq

# ----------------------------------------------------------
#  SocketIO + Flask setup
# ----------------------------------------------------------
sio = socketio.Server()
app = Flask(__name__)

# ----------------------------------------------------------
#  Parameters
# ----------------------------------------------------------
speed_limit = 10
model = None

# ----------------------------------------------------------
#  Virtual traffic-light logic
# ----------------------------------------------------------
current_light = {"color": "green", "last_change": time.time()}

def get_virtual_light():
    now = time.time()
    if now - current_light["last_change"] > 6:
        current_light["color"] = random.choice(["red", "yellow", "green"])
        current_light["last_change"] = now
    return current_light["color"]

# ----------------------------------------------------------
#  Graph for path planning (Dijkstra)
# ----------------------------------------------------------
graph = {
    'A': {'B': 5, 'C': 8},
    'B': {'A': 5, 'D': 7},
    'C': {'A': 8, 'D': 3, 'E': 4},
    'D': {'B': 7, 'C': 3, 'E': 2},
    'E': {'C': 4, 'D': 2}
}

positions = {
    'A': (50, 150),
    'B': (200, 150),
    'C': (100, 220),
    'D': (350, 200),
    'E': (550, 150)
}

def dijkstra(graph, start, goal):
    queue = [(0, start, [])]
    seen = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in seen:
            continue
        path = path + [node]
        seen.add(node)
        if node == goal:
            return (cost, path)
        for n, weight in graph[node].items():
            if n not in seen:
                heapq.heappush(queue, (cost + weight, n, path))
    return (float("inf"), [])

# ----------------------------------------------------------
#  Source → Destination setup
# ----------------------------------------------------------
source, destination = 'A', 'E'
cost, route_nodes = dijkstra(graph, source, destination)
print(f"Selected route: {' -> '.join(route_nodes)}  (cost={cost})")

waypoints = [positions[n] for n in route_nodes]
current_wp = 0
car_pos = list(waypoints[0])
reached = False
blocked_nodes = set()

map_img = np.zeros((300, 600, 3), np.uint8)

# ----------------------------------------------------------
#  Image preprocessing
# ----------------------------------------------------------
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# ----------------------------------------------------------
#  Move car along route
# ----------------------------------------------------------
def update_position():
    global car_pos, current_wp, reached
    if reached or current_wp >= len(waypoints):
        return
    target = np.array(waypoints[current_wp])
    car = np.array(car_pos)
    dir_vec = target - car
    dist = np.linalg.norm(dir_vec)
    if dist < 5:
        current_wp += 1
        if current_wp >= len(waypoints):
            reached = True
            print("✅ Destination reached!")
            return
    else:
        step = 3 * dir_vec / (dist + 1e-5)
        car_pos[0] += step[0]
        car_pos[1] += step[1]

# ----------------------------------------------------------
#  Draw map (full graph + route)
# ----------------------------------------------------------
def draw_map():
    canvas = map_img.copy()
    for a, neighbors in graph.items():
        for b in neighbors:
            cv2.line(canvas, positions[a], positions[b], (100, 100, 100), 2)

    for name, pos in positions.items():
        color = (200, 200, 200)
        if name in route_nodes:
            color = (255, 255, 0)
        if name in blocked_nodes:
            color = (0, 0, 255)
        cv2.circle(canvas, pos, 6, color, -1)
        cv2.putText(canvas, name, (pos[0]-10, pos[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(canvas, tuple(map(int, car_pos)), 8, (0, 255, 0), -1)
    cv2.circle(canvas, positions[destination], 8, (0, 0, 255), -1)

    cv2.imshow("Mini-Map: Route Planning", canvas)
    cv2.waitKey(1)

# ----------------------------------------------------------
#  Detect obstacle visually (simple color/shape detection)
# ----------------------------------------------------------
def detect_obstacle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 1200:  # tune threshold if needed
            return True
    return False

# ----------------------------------------------------------
#  Telemetry handler
# ----------------------------------------------------------
@sio.on('telemetry')
def telemetry(sid, data):
    global model, reached, current_wp, route_nodes, waypoints

    if reached:
        send_control(0, 0)
        return

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.array(image).copy()

    # Virtual light
    color = get_virtual_light()
    if color == "red":
        cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
    elif color == "yellow":
        cv2.circle(image, (50, 50), 15, (0, 255, 255), -1)
    else:
        cv2.circle(image, (50, 50), 15, (0, 255, 0), -1)

    proc_img = img_preprocess(image)
    proc_img = np.array([proc_img])
    steering_angle = float(model.predict(proc_img))

    # Default throttle
    if color == "red":
        throttle = 0.0
    elif color == "yellow":
        throttle = 0.2
    else:
        throttle = 1.0 - speed / speed_limit
        throttle = max(min(throttle, 1.0), 0.0)

    # 🔍 Detect obstacle and re-route dynamically
    if detect_obstacle(image):
        node_to_block = route_nodes[min(current_wp, len(route_nodes)-1)]
        if node_to_block not in blocked_nodes:
            blocked_nodes.add(node_to_block)
            print(f"⚠️ Obstacle detected near {node_to_block}! Re-routing...")
            new_graph = {n:{k:v for k,v in adj.items() if k not in blocked_nodes}
                         for n,adj in graph.items() if n not in blocked_nodes}
            cost, new_path = dijkstra(new_graph, node_to_block, destination)
            if new_path and new_path != route_nodes:
                route_nodes = new_path
                waypoints[:] = [positions[n] for n in route_nodes]
                current_wp = 0
                print(f"➡️ New route: {' -> '.join(route_nodes)} (cost={cost})")
            else:
                print("⚠️ No alternative route found, stopping.")
                reached = True
                send_control(0, 0)
                return

    update_position()
    draw_map()

    print(f"{color.upper():7s} | WP={current_wp}/{len(waypoints)} | "
          f"steer={steering_angle:+.3f} | throttle={throttle:.2f}")

    send_control(steering_angle, throttle)

# ----------------------------------------------------------
#  On connect
# ----------------------------------------------------------
@sio.on('connect')
def connect(sid, environ):
    print('Connected to simulator')
    send_control(0, 0)

# ----------------------------------------------------------
#  Send control
# ----------------------------------------------------------
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# ----------------------------------------------------------
#  Main
# ----------------------------------------------------------
if __name__ == '__main__':
    model = load_model('model/model.h5')
    print("Model loaded successfully.")
    app = socketio.Middleware(sio, app)
    print("Server listening on port 4567 ...")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
