"""plan the spatial relations on topological layer first, then plan the path using A* on metric layer"""
from controller import Supervisor
import numpy as np
import math
import heapq
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, box
from collections import deque

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

# ---------- Grid Parameters ----------
RES = 0.05
ORIGIN = (-1.0, -1.0)
W, H = 40, 40

def world_to_grid(wx, wy):
    return int((wx - ORIGIN[0]) / RES), int((wy - ORIGIN[1]) / RES)

def grid_to_world(col, row):
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES

# -----------RCC8 relation definition ---------------
def rcc8_relation(a, b):
    if a.disjoint(b):
        return "DC"          
    if a.touches(b):
        return "EC"          
    if a.equals(b):
        return "EQ"          
    if a.within(b):
        return "TPP" if a.boundary.intersects(b.boundary) else "NTPP"
    if b.within(a):
        return "TPPi" if a.boundary.intersects(b.boundary) else "NTPPi"
    return "PO"              

# ---------- Define region of environment ----------
DOOR_Y_MIN, DOOR_Y_MAX = -0.15, 0.15   
WALL_X = 0.0                            

REGIONS = {
    "ROOM_LEFT":  box(-1.0, -1.0, WALL_X - 0.025, 1.0),
    "DOOR":       box(WALL_X - 0.025, DOOR_Y_MIN, WALL_X + 0.025, DOOR_Y_MAX),
    "ROOM_RIGHT": box(WALL_X + 0.025, -1.0, 1.0, 1.0),
}

# ---------- construct regions adjacency graph by using RCC8 relations ----------
def build_adjacency(regions):
    """Use RCC8 relations to automatically construct region adjacency graph:EC is considered as passable connection"""
    adj = {name: [] for name in regions}
    names = list(regions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rel = rcc8_relation(regions[a], regions[b])
            if rel == "EC":          # EC = adjacency = could pass through
                adj[a].append(b)
                adj[b].append(a)
            print(f"RCC8({a}, {b}) = {rel}")
    return adj
ADJACENCY = build_adjacency(REGIONS)
print("Region adjacency:", ADJACENCY)

# ---------- Topological layer planning: BFS to find region sequence ----------
def locate_region(x, y, regions):
    """Judge which region a given world coordinate falls into"""
    p = Point(x, y)
    for name, poly in regions.items():
        if poly.contains(p) or poly.touches(p):
            return name
    return None

def topological_plan(start_region, goal_region, adj):
    """In the region adjacency graph, do BFS to find the region sequence"""
    if start_region == goal_region:
        return [start_region]
    queue = deque([[start_region]])
    visited = {start_region}
    while queue:
        path = queue.popleft()
        for nxt in adj[path[-1]]:
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == goal_region:
                return new_path
            visited.add(nxt)
            queue.append(new_path)
    return None

# ---------- A* ----------
def astar(grid, start, goal):
    # grid: H×W, 0=traversable, 1=obstacle; start/goal is (col, row)
    def h(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    # 8 neighborhood
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
            return path[::-1]
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
    return None   

# ---------- obtain robot heading direction----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 rotation matrix, row-major 9 elements
    # projection of robot forward direction in world coordinates, get the angle on x-y plane
    return math.atan2(o[3], o[0])

# ---------- function for adding obstacles into grid    
ROBOT_RADIUS = 0.035 
def add_solid_to_grid(def_name, grid):
    """Mark one Solid obstacle into grid"""
    node = robot.getFromDef(def_name)
    pos = node.getPosition()
    size = node.getField("children").getMFNode(0) \
               .getField("geometry").getSFNode() \
               .getField("size").getSFVec3f()
    hx, hy = size[0]/2, size[1]/2
    # expanded by robot_radius to aviod pass so close the wall
    hx += ROBOT_RADIUS
    hy += ROBOT_RADIUS
    c_min, r_min = world_to_grid(pos[0]-hx, pos[1]-hy)
    c_max, r_max = world_to_grid(pos[0]+hx, pos[1]+hy)
    for r in range(max(0, r_min), min(H, r_max+1)):
        for c in range(max(0, c_min), min(W, c_max+1)):
            grid[r, c] = 1
    # print(f"{def_name}: occupied col[{c_min}~{c_max}] row[{r_min}~{r_max}]")

# ---------- planning ----------
grid = np.zeros((H, W))          
#import a base obstacle（wall）
add_solid_to_grid("WALL1", grid)
add_solid_to_grid("WALL2", grid)

# ---------- Metric layer planning: segment A* to find path in grid, two level planning ----------
def region_transition_point(region_name, regions):
    """take region name and return its representative point (centroid) as a transition target"""
    c = regions[region_name].centroid
    return (c.x, c.y)
# two level planning main flow
ep = epuck.getPosition()
rb = red_box.getPosition()

start_region = locate_region(ep[0], ep[1], REGIONS)
goal_region  = locate_region(rb[0], rb[1], REGIONS)
print(f"Start in {start_region}, goal in {goal_region}")

region_seq = topological_plan(start_region, goal_region, ADJACENCY)
print("Topological plan:", " -> ".join(region_seq))

# start -> mass point of each region -> goal
metric_targets = [(ep[0], ep[1])]
for rname in region_seq[1:-1]:                    # pass region(DOOR here)
    metric_targets.append(region_transition_point(rname, REGIONS))
metric_targets.append((rb[0], rb[1]))

# segmented A*, get together for completing routine
full_path = []
for i in range(len(metric_targets) - 1):
    s = world_to_grid(*metric_targets[i])
    g = world_to_grid(*metric_targets[i + 1])
    seg = astar(grid, s, g)
    if seg is None:
        print(f"Segment {i} planning failed")
        full_path = None
        break
    if full_path:
        seg = seg[1:]        # delete duplicate waypoints
    full_path.extend(seg)

if full_path is None:
    waypoints = []
else:
    waypoints = [grid_to_world(c, r) for (c, r) in full_path]
    print(f"Totally {len(waypoints)} checkpoints through the path")

wp_index = 0
REACH = 0.06        
GOAL_REACH = 0.08   
trajectory = [] # to record routine that robot has passed

while robot.step(timestep) != -1:
    if not waypoints or wp_index >= len(waypoints):
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        continue

    ep = epuck.getPosition()
    trajectory.append((ep[0], ep[1]))
    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    dist = math.hypot(dx, dy)

    if dist < REACH:
        wp_index += 1
        if wp_index >= len(waypoints):
            print(">>> Arrived Red_Box nearby")
            # ---------- export grid PNG ----------
            fig, ax = plt.subplots(figsize=(7, 7))
            cmap = matplotlib.colors.ListedColormap(["white", "dimgray"])
            ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=1)
            # 2. trajectory of A* planned path(dashed)
            wp_grid = [world_to_grid(p[0], p[1]) for p in waypoints]
            wp_x = [g[0] for g in wp_grid]
            wp_y = [g[1] for g in wp_grid]
            ax.plot(wp_x, wp_y, "b--", linewidth=1.5,
                    marker="o", markersize=4, label="A* planned path")
            # 3. actual robot trajectory(solid line)
            tr_grid = [world_to_grid(p[0], p[1]) for p in trajectory]
            tr_x = [g[0] for g in tr_grid]
            tr_y = [g[1] for g in tr_grid]
            ax.plot(tr_x, tr_y, "g-", linewidth=2, label="Actual trajectory")

            # grid line for each box
            ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax.grid(which="minor", color="lightgray", linewidth=0.4)

            # marked main scaler per 5 boxes
            ax.set_xticks(np.arange(0, W, 5))
            ax.set_yticks(np.arange(0, H, 5))

            sc, sr = world_to_grid(trajectory[0][0], trajectory[0][1])
            gc, gr = world_to_grid(rb[0], rb[1])
            ax.plot(sc, sr, "go", markersize=10, label="Start")
            ax.plot(gc, gr, "rs", markersize=10, label="Goal")

            ax.set_title("Pass Door Routine")
            ax.set_xlabel("col (x)")
            ax.set_ylabel("row (y)")
            ax.legend(loc="upper left", fontsize=9)

            plt.savefig("pass_door_routine.png", dpi=130, bbox_inches="tight")
            print(">>> exported pass_door_routine.png")
        continue

    # calculate the required turning angle: target direction - current heading
    target_angle = math.atan2(dy, dx)
    heading = get_heading()
    err = target_angle - heading
    # normalize to [-pi, pi]
    err = math.atan2(math.sin(err), math.cos(err))

    # easy control: rotate when heading deviation is large, go straight when deviation is small
    if abs(err) > 0.3:
        turn = 2.0 if err > 0 else -2.0
        left.setVelocity(-turn)
        right.setVelocity(turn)
    else:
        # go straight + slight correction
        base = 0.5 * MAX_SPEED
        corr = 2.0 * err
        left.setVelocity(base - corr)
        right.setVelocity(base + corr)
