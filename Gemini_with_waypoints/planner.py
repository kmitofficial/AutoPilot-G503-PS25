#!/usr/bin/env python3
"""
planner.py
- Build a graph from `cords.txt` (or use `list.txt` if present), compute a shortest path
  between start and goal nodes, and publish the planned waypoint sequence to MQTT
  topics consumed by `rover.py` and `gemini.py` (topic `rover/waypoints`).

Usage:
  python planner.py --cords cords.txt --start 0 --goal 12 --broker 10.208.218.104

If --start or --goal are omitted, planner will use the first and last coordinates.
"""
import os
import json
import math
import argparse
from time import sleep

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

from get_graph import haversine_distance


def read_coords_robust(filename):
    coords = {}
    idx = 0
    float_re = r"[-+]?\d*\.\d+|\d+"
    import re
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # replace commas with spaces
            s2 = s.replace(',', ' ')
            toks = s2.split()
            numbers = []
            for t in toks:
                try:
                    numbers.append(float(t))
                except Exception:
                    found = re.findall(float_re, t)
                    for fs in found:
                        try:
                            numbers.append(float(fs))
                        except Exception:
                            pass
            if len(numbers) >= 3:
                lat, lon, alt = numbers[:3]
                coords[idx] = (lat, lon, alt)
                idx += 1
            else:
                print("⚠️ skipping unparsable line:", line.strip())
    return coords


def read_connections(filename):
    connections = {}
    if not os.path.exists(filename):
        return connections
    with open(filename, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            a, rest = line.strip().split(':', 1)
            a = int(a.strip())
            neighbors = [int(x.strip()) for x in rest.split(',') if x.strip()]
            connections[a] = neighbors
    return connections


def build_knn_connections(coords, k=3):
    # Connect each node to its k nearest neighbors and also to next/prev sequentially
    nodes = list(coords.keys())
    conns = {n: set() for n in nodes}
    for i in nodes:
        # sequential neighbors
        if (i + 1) in coords:
            conns[i].add(i + 1); conns[i + 1].add(i)
    # KNN
    for i in nodes:
        dists = []
        for j in nodes:
            if i == j:
                continue
            d = haversine_distance(coords[i], coords[j])
            dists.append((d, j))
        dists.sort()
        for _, j in dists[:k]:
            conns[i].add(j)
            conns[j].add(i)
    # convert sets to lists
    return {n: sorted(list(neis)) for n, neis in conns.items()}


def create_weighted_adj(coords, connections):
    adj = {}
    for n, neis in connections.items():
        adj[n] = {}
        for m in neis:
            if m not in coords:
                continue
            adj[n][m] = round(haversine_distance(coords[n], coords[m]), 2)
    return adj


def dijkstra(adj, start, goal):
    import heapq
    dist = {n: float('inf') for n in adj}
    prev = {n: None for n in adj}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if node == goal:
            break
        if d > dist[node]:
            continue
        for nbr, w in adj.get(node, {}).items():
            nd = d + w
            if nd < dist.get(nbr, float('inf')):
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(pq, (nd, nbr))
    if dist.get(goal, float('inf')) == float('inf'):
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def publish_waypoints(broker, port, topic, coords, path):
    if mqtt is None:
        print("⚠️ paho-mqtt not installed; cannot publish waypoints")
        return False
    client = mqtt.Client()
    try:
        client.connect(broker, port, 60)
    except Exception as e:
        print("⚠️ Could not connect to broker:", e)
        return False
    payload = {'waypoints': [{'lat': coords[n][0], 'lon': coords[n][1], 'alt': coords[n][2]} for n in path]}
    client.publish(topic, json.dumps(payload), qos=1)
    print(f"➡️ Published {len(path)} waypoints to {topic} on {broker}:{port}")
    client.disconnect()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cords', default='cords.txt')
    p.add_argument('--list', default='list.txt')
    p.add_argument('--start', type=int, help='start node index')
    p.add_argument('--goal', type=int, help='goal node index')
    p.add_argument('--broker', default='localhost')
    p.add_argument('--port', type=int, default=1883)
    p.add_argument('--k', type=int, default=3, help='k for k-nearest neighbors when building graph')
    p.add_argument('--topic', default='rover/waypoints')
    args = p.parse_args()

    coords = read_coords_robust(args.cords)
    if not coords:
        print('No coordinates found in', args.cords); return

    connections = read_connections(args.list)
    if not connections:
        connections = build_knn_connections(coords, k=args.k)
        print(f"Built knn connections with k={args.k}")
    else:
        print(f"Loaded explicit connections from {args.list}")

    adj = create_weighted_adj(coords, connections)

    # choose start/goal
    nodes = sorted(coords.keys())
    start = args.start if args.start is not None else nodes[0]
    goal = args.goal if args.goal is not None else nodes[-1]

    if start not in adj or goal not in adj:
        print('Start/goal not in graph nodes')
        return

    path = dijkstra(adj, start, goal)
    if not path:
        print('No path found between', start, goal); return

    print('Path nodes:', path)
    # save planned path
    with open('planned_path.json', 'w') as f:
        json.dump({'path': path, 'coords': {n: coords[n] for n in path}}, f, indent=2)

    # publish to MQTT so rover and gemini receive it
    publish_waypoints(args.broker, args.port, args.topic, coords, path)


if __name__ == '__main__':
    main()
