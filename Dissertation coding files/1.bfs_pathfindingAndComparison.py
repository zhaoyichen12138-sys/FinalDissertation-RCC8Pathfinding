""" bfs and astar pathfinding comparison """
from collections import deque
import math
import time

def bfs(grid, start, goal):
    H, W = grid.shape
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    visited = {start}
    came = {}
    queue = deque([start])
    expanded = 0   # number of nodes actually popped from the queue and processed

    while queue:
        cur = queue.popleft()          # FIFO：deal with the first element in the queue
        expanded += 1
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path = path[::-1]
            if return_stats:
                return path, {"expanded nodes": expanded}
            return path

        cx, cy = cur
        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue
            nxt = (nx, ny)
            if nxt in visited:          # no comparison of g[] costs, just enqueue if not visited
                continue
            visited.add(nxt)
            came[nxt] = cur
            queue.append(nxt)

    if return_stats:
        return None, {"expanded nodes": expanded}
    return None


def path_length_meters(waypoints):
    """calculate the total true Euclidean distance of a path (in meters), for fair comparison with A*.
    Under 8-connected neighborhood, BFS's "fewest hops" doesn't equal "shortest real distance."""
    total = 0.0
    for i in range(1, len(waypoints)):
        x1, y1 = waypoints[i - 1]
        x2, y2 = waypoints[i]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def run_comparison(astar_fn, grid, start, goal, grid_to_world):
    """
    Comparesion parameters:
    - time spent (seconds)
    - number of expanded/visited nodes (search efficiency)
    - true length of the path (meters)
    """
    import inspect
    results = {}
    for name, fn in [("A*", astar_fn), ("BFS", bfs)]:
        t0 = time.perf_counter()
        supports_stats = "return_stats" in inspect.signature(fn).parameters
        if supports_stats:
            path, stats = fn(grid, start, goal, return_stats=True)
            expanded = stats["expanded nodes"]
        else:
            path = fn(grid, start, goal)
            expanded = "N/A(need add expanded counter to astar() to display)"
        elapsed = time.perf_counter() - t0

        if path is None:
            results[name] = {"path": None, "time": elapsed}
            continue

        waypoints = [grid_to_world(c, r) for (c, r) in path]
        results[name] = {
            "path": path,
            "waypoints": waypoints,
            "time": elapsed,
            "expanded": expanded,
            "path_nodes": len(path),
            "length_m": path_length_meters(waypoints),
        }

    for name, r in results.items():
        if r["path"] is None:
            print(f"[{name}] did not find a path, took {r['time']*1000:.2f} ms")
        else:
            print(f"[{name}] took {r['time']*1000:.2f} ms | "
                  f"expanded(actual search nodes) {r['expanded']} | "
                  f"path nodes {r['path_nodes']} | true length {r['length_m']:.3f} m")

    return results
