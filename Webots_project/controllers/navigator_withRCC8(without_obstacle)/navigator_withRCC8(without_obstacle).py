"""navigator: A* combined with RCC8 relations"""
from controller import Supervisor
from shapely.geometry import Point, Polygon
from bfs_pathfinding import bfs, run_comparison
import numpy as np
import math
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")

# get two wheel motors, set to velocity control mode
left = robot.getDevice("left wheel motor")
right = robot.getDevice("right wheel motor")
left.setPosition(float("inf"))
right.setPosition(float("inf"))
left.setVelocity(0.0)
right.setVelocity(0.0)

MAX_SPEED = 6.28   

# ---------- Grid parameters ----------
RES = 0.05
ORIGIN = (-1.0, -1.0)
W, H = 40, 40

def world_to_grid(wx, wy):
    return int((wx - ORIGIN[0]) / RES), int((wy - ORIGIN[1]) / RES)

def grid_to_world(col, row):
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES

# ---------- A* ----------
def astar(grid, start, goal, return_stats=False):
    # grid: H×W, 0=traversable, 1=obstacle; start/goal is (col, row)
    def h(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    # 8 neighbors
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    expanded = 0 # to show difference between astar and bfs searching
    while open_set:
        _, cur = heapq.heappop(open_set)
        expanded += 1 # astar and bfs comparison index
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
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue
            step = math.hypot(dx, dy)
            ng = g[cur] + step
            nxt = (nx, ny)
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + h(nxt, goal), nxt))
    
    if return_stats:
        return None, {"expanded nodes": expanded}
    return None   

# ---------- get robot heading ----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 rotation matrix, row-major order 9 numbers
    # the projection of the robot's front in the world coordinate system, get the angle on the x-y plane
    return math.atan2(o[3], o[0])

# -----------RCC8 relation definition ---------------
def rcc8_relation(a, b):
    """returns the basic RCC8 relation between regions a and b"""
    if a.disjoint(b):
        return "DC"          
    if a.touches(b):
        return "EC"          # externally connected
    if a.equals(b):
        return "EQ"          
    if a.within(b):
        return "TPP" if a.boundary.intersects(b.boundary) else "NTPP"
    if b.within(a):
        return "TPPi" if a.boundary.intersects(b.boundary) else "NTPPi"
    return "PO"              # overlap

# ---------- define forbidden zone(logic constraint) ----------
FORBIDDEN_ZONE_1 = Polygon([
    (0.10, -0.10),
    (0.35, -0.10),
    (0.35,  0.20),
    (0.10,  0.20),
])
FORBIDDEN_ZONE_2 = Polygon([
    (-0.50, -0.70),
    (-0.10, -0.70),
    (-0.10,  -0.20),
    (-0.50,  -0.20),
])
FORBIDDEN_ZONES = [FORBIDDEN_ZONE_1, FORBIDDEN_ZONE_2]

ROBOT_RADIUS = 0.035  

# safety margin for planning (meters).
SAFETY_MARGIN = 0.03
PLAN_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN

def robot_footprint(x, y):
    """approximate the robot as a circular region"""
    return Point(x, y).buffer(ROBOT_RADIUS)

grid = np.zeros((H, W))        
# ---------------- rasterising forbidden zone ------------
forbidden_mask = np.zeros((H, W))

for r in range(H):
    for c in range(W):
        wx, wy = grid_to_world(c, r)
        cell = Point(wx, wy).buffer(PLAN_RADIUS)   
        for zone in FORBIDDEN_ZONES:
            rel = rcc8_relation(cell, zone)
            if rel not in ("DC", "EC"):
                forbidden_mask[r, c] = 1
                grid[r, c] = 1       
                break
print(f"Forbidden rasterising finished, occupied {int(forbidden_mask.sum())} grids")

ep = epuck.getPosition()
rb = red_box.getPosition()
start = world_to_grid(ep[0], ep[1])
goal = world_to_grid(rb[0], rb[1])

comparison = run_comparison(astar, grid, start, goal, grid_to_world)
path = astar(grid, start, goal)

# routine planned by BFS:only for comparison plotting, robot still move by A*'s path
bfs_path, bfs_stats = bfs(grid, start, goal, return_stats=True)
if bfs_path is None:
    print("BFS cannot find the path")
    bfs_waypoints = []
else:
    bfs_waypoints = [grid_to_world(c, r) for (c, r) in bfs_path]
    print(f"BFS path: {len(bfs_waypoints)} checkpoints, "
          f"expanded {bfs_stats['expanded nodes']} nodes")

if path is None:
    print("Cannot find the path")
    waypoints = []
else:
    waypoints = [grid_to_world(c, r) for (c, r) in path]
    print(f"Totally {len(waypoints)} checkpoints through the path")

wp_index = 0
# must less than RES(0.05), otherwise the robot will switch to the next waypoint before reaching the current one, causing it to deviate from the planned path
REACH = 0.03        
GOAL_REACH = 0.08   

trajectory = [] # to record routine that robot has passed
violations = []  # record the locations and relationships that violate constraints, used for marking on the graph
last_relation = [None] * len(FORBIDDEN_ZONES)  # the relationship from the previous step, used to only print when it changes


# ----------------mainloop --------------------
while robot.step(timestep) != -1:
    if not waypoints or wp_index >= len(waypoints):
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        continue

    ep = epuck.getPosition()
    trajectory.append((ep[0], ep[1]))
    # judge RCC8 relationship between robot and forbidden zone in real-time
    fp = robot_footprint(ep[0], ep[1])
    for i, zone in enumerate(FORBIDDEN_ZONES):
        rel = rcc8_relation(fp, zone)
        if rel not in ("DC", "EC"):
            violations.append((ep[0], ep[1], i + 1, rel))
            # only print once when the relationship changes, to avoid flooding the console with messages
            if rel != last_relation[i]:
                print(f"!!! Violate constraint: relation with zone {i+1} "
                      f"is {rel} at ({ep[0]:.3f}, {ep[1]:.3f})")
        last_relation[i] = rel
    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    dist = math.hypot(dx, dy)

    # arrived at current waypoint, switch to the next
    if dist < REACH:
        wp_index += 1
        if wp_index >= len(waypoints):
            print(">>> Arrived Red_Box nearby")
            # ----- routine and forbidden zone plot -----
            fig, ax = plt.subplots(figsize=(8, 8))

            # 1. forbidden zone(red boundary)
            for i, zone in enumerate(FORBIDDEN_ZONES):
                zone_coords = list(zone.exterior.coords)
                ax.add_patch(MplPolygon(zone_coords, closed=True,
                                        facecolor="red", alpha=0.15,
                                        edgecolor="red", linewidth=2.5,
                                        label="Forbidden zone (RCC8)" if i == 0 else None))

            # 2. boundary of forbidden zone inflated by robot radius (dashed line)
            for i, zone in enumerate(FORBIDDEN_ZONES):
                infl = list(zone.buffer(ROBOT_RADIUS).exterior.coords)
                ax.add_patch(MplPolygon(infl, closed=True,
                                        facecolor="none",
                                        edgecolor="red", linewidth=1.2,
                                        linestyle=":",
                                        label="Zone inflated by robot radius" if i == 0 else None))

            # 3. actual trajectory of the robot (solid line)
            tr_x = [p[0] for p in trajectory]
            tr_y = [p[1] for p in trajectory]
            ax.plot(tr_x, tr_y, "g-", linewidth=2, label="Actual trajectory")

            # 4. constraint violations (positions where PO/TPP violations occur)
            if violations:
                vx = [v[0] for v in violations]
                vy = [v[1] for v in violations]
                ax.plot(vx, vy, "rx", markersize=6, linestyle="none",
                        label=f"Constraint violations ({len(violations)} steps)")

            # 5. start and stop points
            ax.plot(tr_x[0], tr_y[0], "go", markersize=12, label="Start")
            ax.plot(rb[0], rb[1], "rs", markersize=12, label="Target (red box)")

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Robot avoids forbidden zone")
            ax.legend(loc="upper left", fontsize=9)
            plt.savefig("avoids_forbiddenZone_path.png", dpi=150, bbox_inches="tight")
            print(">>> exported avoids_forbiddenZone_path.png")
        continue

    # calculate the required turn: target direction - current heading
    target_angle = math.atan2(dy, dx)
    heading = get_heading()
    err = target_angle - heading
    # normalize to [-pi, pi]
    err = math.atan2(math.sin(err), math.cos(err))

    # easy control: large heading error results in turning on the spot, small error leads to straight movement
    if abs(err) > 0.3:
        turn = 2.0 if err > 0 else -2.0
        left.setVelocity(-turn)
        right.setVelocity(turn)
    else:
        # straight movement + minor correction
        base = 0.5 * MAX_SPEED
        corr = 2.0 * err
        left.setVelocity(base - corr)
        right.setVelocity(base + corr)
