# server/path_planner_offline.py
import math
import heapq
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_coords(cords_path=None):
    if cords_path is None:
        cords_path = os.path.join(BASE_DIR, "cords.txt")
    coords = {}
    with open(cords_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            node = int(parts[0])
            lat, lon, alt = map(float, parts[1:4])
            coords[node] = (lat, lon, alt)
    return coords

def load_connections(list_path=None):
    if list_path is None:
        list_path = os.path.join(BASE_DIR, "list.txt")
    con = {}
    with open(list_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ':' not in ln: continue
            node, neighbors = ln.split(':', 1)
            node = int(node.strip())
            nbrs = [int(x.strip()) for x in neighbors.split(',') if x.strip()]
            con[node] = nbrs
    return con

def build_weighted_adj(coords, conns):
    adj = {}
    for n, nbrs in conns.items():
        adj[n] = {}
        for nb in nbrs:
            if nb not in coords:
                continue
            d = haversine_m(coords[n][0], coords[n][1], coords[nb][0], coords[nb][1])
            adj[n][nb] = d
    return adj

def nearest_node(lat, lon, coords):
    best = None
    bestd = float('inf')
    for node, (plat, plon, _) in coords.items():
        d = haversine_m(lat, lon, plat, plon)
        if d < bestd:
            bestd = d
            best = node
    return best, bestd

def dijkstra(adj, start, goal):
    # adj: {node: {nbr: weight}}
    pq = [(0, start)]
    dist = {start: 0}
    parent = {}
    visited = set()
    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited: 
            continue
        visited.add(u)
        if u == goal:
            break
        for v, w in adj.get(u, {}).items():
            nd = cost + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    if goal not in dist:
        return None
    # reconstruct
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path, dist[goal]

def path_nodes_to_coords(path_nodes, coords):
    return [[coords[n][0], coords[n][1]] for n in path_nodes]

# Convenience function: given start lat/lon and dest node index, return waypoint list
def plan_from_coords(start_lat, start_lon, dest_node_index, cords_path=None, list_path=None):
    coords = load_coords(cords_path)
    conns = load_connections(list_path)
    adj = build_weighted_adj(coords, conns)
    start_node, start_d = nearest_node(start_lat, start_lon, coords)
    if start_node is None:
        raise RuntimeError("No start node found")
    if dest_node_index not in coords:
        raise RuntimeError("Destination node not in coords")
    res = dijkstra(adj, start_node, dest_node_index)
    if res is None:
        raise RuntimeError("No path found")
    path_nodes, cost_m = res
    waypoints = path_nodes_to_coords(path_nodes, coords)
    return waypoints, {"cost_m": cost_m, "start_node": start_node}
