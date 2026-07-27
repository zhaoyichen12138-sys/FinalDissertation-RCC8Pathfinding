"""navigator: A* 规划 + e-puck 差速轮控制,走到红盒子"""
from controller import Supervisor
import numpy as np
import math
import heapq
import matplotlib
matplotlib.use("Agg")          # 无窗口后台绘图,存文件用
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, box
from collections import deque

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")

# 拿到两个轮子电机,设成速度控制模式
left = robot.getDevice("left wheel motor")
right = robot.getDevice("right wheel motor")
left.setPosition(float("inf"))
right.setPosition(float("inf"))
left.setVelocity(0.0)
right.setVelocity(0.0)

MAX_SPEED = 6.28   # e-puck 电机上限约 6.28 rad/s

# ---------- 栅格参数(和上一步一致) ----------
RES = 0.05
ORIGIN = (-1.0, -1.0)
W, H = 40, 40

def world_to_grid(wx, wy):
    return int((wx - ORIGIN[0]) / RES), int((wy - ORIGIN[1]) / RES)

def grid_to_world(col, row):
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES

# -----------RCC8 relation definition ---------------
def rcc8_relation(a, b):
    """返回区域 a 与区域 b 的 RCC8 基本关系"""
    if a.disjoint(b):
        return "DC"          # 相离
    if a.touches(b):
        return "EC"          # 外切(仅边界接触)
    if a.equals(b):
        return "EQ"          # 相等
    if a.within(b):
        # 区分切内含与非切内含:边界是否接触
        return "TPP" if a.boundary.intersects(b.boundary) else "NTPP"
    if b.within(a):
        return "TPPi" if a.boundary.intersects(b.boundary) else "NTPPi"
    return "PO"              # 部分重叠

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
            # print(f"RCC8({a}, {b}) = {rel}")
    return adj
ADJACENCY = build_adjacency(REGIONS)
# print("Region adjacency:", ADJACENCY)

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
    # grid: H×W, 0=可走 1=障碍; start/goal 是 (col,row)
    def h(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    # 8 邻域
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_set:
        _, cur = heapq.heappop(open_set)
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
    return None   # 无路

# ---------- obtain robot heading direction(绕竖直 z 轴的偏航角)----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 旋转矩阵,行主序 9 个数
    # 机器人前方在世界坐标的投影,取 x-y 平面上的角度
    return math.atan2(o[3], o[0])

# ---------- function for adding obstacles into grid    
ROBOT_RADIUS = 0.035  # e-puck 半径约 3.5cm
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

# ---------- 规划一次 ----------
UNKNOWN, FREE, OCCUPIED = -1, 0, 1
known_map = np.full((H, W), UNKNOWN)   # robot known map, initial all unknown(updating with exploration)

# -------- real-world map (Only for local perception searching, robot cannot use it for planning directly)
true_map = np.zeros((H, W))
add_solid_to_grid("WALL1", true_map)
add_solid_to_grid("WALL2", true_map)

SENSOR_RANGE = 0.25   # perception radius(meter), around 5 grid cells

# ---------- perception function: show true values within perception radius on known map ----------
def sense(robot_x, robot_y, known_map, true_map):
    """show true values within perception radius on known map, return whether new obstacle found"""
    rc, rr = world_to_grid(robot_x, robot_y)
    radius_cells = int(SENSOR_RANGE / RES)
    found_new_obstacle = False

    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            r, c = rr + dr, rc + dc
            if not (0 <= r < H and 0 <= c < W):
                continue
            if math.hypot(dr, dc) > radius_cells:
                continue          # circle perception radius
            if known_map[r, c] == UNKNOWN:
                known_map[r, c] = true_map[r, c]
                if true_map[r, c] == OCCUPIED:
                    found_new_obstacle = True
    return found_new_obstacle

def try_detect_target(robot_x, robot_y):
    """check if the red box is within the sensor range; if so, return its grid coordinates, otherwise None"""
    rb = red_box.getPosition()
    dist = math.hypot(rb[0] - robot_x, rb[1] - robot_y)
    if dist <= SENSOR_RANGE:
        return world_to_grid(rb[0], rb[1])
    return None

def find_nearest_frontier(known_map, robot_rc):
    """find the nearest frontier cell (unknown cell adjacent to known free space)"""
    frontiers = []
    for r in range(H):
        for c in range(W):
            if known_map[r, c] != FREE:
                continue
            # check if any of the four neighbors is an unknown cell
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and known_map[nr, nc] == UNKNOWN:
                    frontiers.append((c, r))
                    break
    if not frontiers:
        return None
    # choose the nearest one to the robot
    rc, rr = robot_rc
    frontiers.sort(key=lambda f: math.hypot(f[0]-rc, f[1]-rr))
    return frontiers[0]

def planning_map(known_map):
    """treat as accessble area if unknown, and do A* planning"""
    m = known_map.copy()
    m[m == UNKNOWN] = FREE
    return m

wp_index = 0
REACH = 0.06        # the distance threshold to reach a waypoint (m)
GOAL_REACH = 0.08   # the distance threshold to reach the goal
# REPLAN_INTERVAL = 20      # enforce replan every 20 steps
step_count = 0
waypoints = []
replan_count = 0
trajectory = [] # to record routine that robot has passed
goal =  None
goal_found = False

while robot.step(timestep) != -1:
    ep = epuck.getPosition()
    trajectory.append((ep[0], ep[1]))
    step_count += 1

    # ---- perception ----
    found_new = sense(ep[0], ep[1], known_map, true_map)

    # ---- try to detect target ----
    if not goal_found:
        detected = try_detect_target(ep[0], ep[1])
        if detected:
            goal = detected
            goal_found = True
            waypoints = []          # empty the old exploration path, force replanning towards the target
            print(f">>> target detected! Located at grid {goal}")

    # ---- decide where to go next ----
    need_replan = (found_new or not waypoints
                   or wp_index >= len(waypoints))
                #    or step_count % REPLAN_INTERVAL == 0)

    if need_replan:
        start = world_to_grid(ep[0], ep[1])
        pmap = planning_map(known_map)

        if goal_found:
            target_cell = goal                          
        else:
            target_cell = find_nearest_frontier(known_map, start)   # explore unknown area if goal not found yet

        if target_cell is None:
            print(">>> map exploration complete, no frontiers left")
            left.setVelocity(0.0); right.setVelocity(0.0)
            break

        new_path = astar(pmap, start, target_cell)
        if new_path:
            waypoints = [grid_to_world(c, r) for (c, r) in new_path]
            wp_index = 1 if len(waypoints) > 1 else 0
            replan_count += 1

    # ---- judge whether goal is reached ----
    if goal_found:
        rb = red_box.getPosition()
        dist_to_goal = math.hypot(rb[0] - ep[0], rb[1] - ep[1])
        if dist_to_goal < GOAL_REACH:
            left.setVelocity(0.0)
            right.setVelocity(0.0)
            print(f">>> Arrived. Replanned {replan_count} times.")
            break
    
    # ---- execute: move towards current waypoint ----
    if not waypoints or wp_index >= len(waypoints):
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        continue

    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    if math.hypot(dx, dy) < REACH:
        wp_index += 1
        continue

    target_angle = math.atan2(dy, dx)
    err = math.atan2(math.sin(target_angle - get_heading()),
                     math.cos(target_angle - get_heading()))
    if abs(err) > 0.3:
        turn = 2.0 if err > 0 else -2.0
        left.setVelocity(-turn)
        right.setVelocity(turn)
    else:
        base = 0.5 * MAX_SPEED
        corr = 2.0 * err
        left.setVelocity(base - corr)
        right.setVelocity(base + corr)

# ---------- export grid PNG ----------
fig, ax = plt.subplots(figsize=(7, 7))
cmap = matplotlib.colors.ListedColormap(["lightgray", "white", "dimgray"]) # three colors for unknown, free, occupied
ax.imshow(known_map + 1, origin="lower", cmap=cmap, vmin=0, vmax=2)
# # 2. A* planned trajectory(virtual line)
# wp_grid = [world_to_grid(p[0], p[1]) for p in waypoints]
# wp_x = [g[0] for g in wp_grid]
# wp_y = [g[1] for g in wp_grid]
# ax.plot(wp_x, wp_y, "b--", linewidth=1.5,
#         marker="o", markersize=4, label="A* planned trajectory")
# 3. robot actual trajectory(实线)
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

ax.set_title("Robot`s Routine")
ax.set_xlabel("col (x)")
ax.set_ylabel("row (y)")
ax.legend(loc="upper left", fontsize=9)

plt.savefig("robot_routine.png", dpi=130, bbox_inches="tight")
print(">>> exported robot_routine.png")
