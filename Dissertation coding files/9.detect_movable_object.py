"""local perception with memory: the robot can remember the known map and update it with new perception, and do A* planning on the known map"""
from controller import Supervisor
import os
import numpy as np
import math
import heapq
import matplotlib
matplotlib.use("Agg")          # non-windows plot, use to save files
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, Polygon, box
from collections import deque
import itertools

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")

# two wheels, set to velocity control mode
left = robot.getDevice("left wheel motor")
right = robot.getDevice("right wheel motor")
left.setPosition(float("inf"))
right.setPosition(float("inf"))
left.setVelocity(0.0)
right.setVelocity(0.0)

MAX_SPEED = 6.28   # e-puck`s` max speed is around 6.28 rad/s

# ---------- grid parameters ----------
RES = 0.05 # meter per grid cell
ORIGIN = (-1.0, -1.0)
W, H = 40, 40

def world_to_grid(wx, wy):
    return int((wx - ORIGIN[0]) / RES), int((wy - ORIGIN[1]) / RES)

def grid_to_world(col, row):
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES

# -----------RCC8 relation definition ---------------
def rcc8_relation_tol(a, b, tol=0.02):
    """带容差的 RCC8:距离小于 tol 视为 EC,避免离散跳变错过相切"""
    if a.intersects(b):
        # 已经相交,细分 PO / 内含
        if a.equals(b): return "EQ"
        if a.within(b): return "TPP" if a.boundary.intersects(b.boundary) else "NTPP"
        if b.within(a): return "TPPi" if a.boundary.intersects(b.boundary) else "NTPPi"
        if a.touches(b): return "EC"
        return "PO"
    # 未相交:看距离是否在容差内
    if a.distance(b) <= tol:
        return "EC"          # 足够近,视为刚好接触
    return "DC"

# ---------- Define region of environment ----------
DOOR_Y_MIN, DOOR_Y_MAX = -0.15, 0.15   
WALL_X = 0.0                            

REGIONS = {
    "ROOM_LEFT":  box(-1.0, -1.0, WALL_X - 0.025, 1.0),
    "DOOR":       box(WALL_X - 0.025, DOOR_Y_MIN, WALL_X + 0.025, DOOR_Y_MAX),
    "ROOM_RIGHT": box(WALL_X + 0.025, -1.0, 1.0, 1.0),
}

FORBIDDEN_ZONE1 = Polygon([
    (-0.80, 0.70),
    (-0.40, 0.70),
    (-0.40,  0.40),
    (-0.80,  0.40),
])
FORBIDDEN_ZONE2 = Polygon([
    (0.40, -0.30),
    (0.70, -0.30),
    (0.70, -0.60),
    (0.40, -0.60),
])
FORBIDDEN_ZONES = [FORBIDDEN_ZONE1, FORBIDDEN_ZONE2]
ROBOT_RADIUS = 0.035   # e-puck 半径约 3.5cm
#SAFETY_MARGIN = 0.03
PLAN_RADIUS = ROBOT_RADIUS

def robot_footprint(x, y):
    """把机器人近似成一个圆形区域"""
    return Point(x, y).buffer(ROBOT_RADIUS)

def make_box_region(pos, half=0.05):
    """similar to the function for creating a box region"""
    return box(pos[0]-half, pos[1]-half, pos[0]+half, pos[1]+half)
# monitor objects: forbidden zone, door, obstacles

last_relations = {}   # records the RCC8 relations for each object in last time

def monitor_rcc8(robot_fp, targets):
    """
    robot_fp: footprint of robot at that time(shapely circle)
    targets: {name: shapely polygon}
    return the list of changed relations and update last_relations
    """
    changes = []
    for name, region in targets.items():
        rel = rcc8_relation_tol(robot_fp, region)
        if last_relations.get(name) != rel:
            prev = last_relations.get(name, "None")
            changes.append((name, prev, rel))
            last_relations[name] = rel
    return changes

# ---------- construct regions adjacency graph by using RCC8 relations ----------
def build_adjacency(regions):
    """Use RCC8 relations to automatically construct region adjacency graph:EC is considered as passable connection"""
    adj = {name: [] for name in regions}
    names = list(regions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rel = rcc8_relation_tol(regions[a], regions[b])
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
    # grid: H×W, start/goal is (col,row)
    def h(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    # 8 neighbors: 4 cardinal + 4 diagonal
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
    return None

# ---------- obtain robot heading direction(绕竖直 z 轴的偏航角)----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 旋转矩阵,行主序 9 个数
    # 机器人前方在世界坐标的投影,取 x-y 平面上的角度
    return math.atan2(o[3], o[0])

# ---------- function for adding obstacles into grid    
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

# ---------- Movable object recognition -----------
MOVABLE_MASS_THRESHOLD = 0.15  # kg, if mass < threshold, consider it as movable object
def get_object_mass(def_name):
    """Get the mass of a node"""
    node = robot.getFromDef(def_name)
    physics_field = node.getField("physics")
    physics = physics_field.getSFNode()
    if physics is None:
        return None
    return physics.getField("mass").getSFFloat()

# --------- detect whether has space to move the object by using RCC8 relations ---------
def has_push_space(obj_pos, push_dir, known_map):
    """Check if there is enough free space in the push direction to move the object"""
    # the point 0.15m ahead of the object
    probe_x = obj_pos[0] + push_dir[0] * 0.15
    probe_y = obj_pos[1] + push_dir[1] * 0.15
    c, r = world_to_grid(probe_x, probe_y)
    if not (0 <= c < W and 0 <= r < H): # out of grid, no space to push
        return False                    
    return known_map[r, c] != OCCUPIED

# --------- fully judgement of movable detection: mass + push space ---------
def is_pushable(def_name, robot_pos, known_map):
    mass = get_object_mass(def_name)
    if mass is None or mass >= MOVABLE_MASS_THRESHOLD:
        return False                    # 
    obj_pos = robot.getFromDef(def_name).getPosition()
    # push direction = from robot to object
    dx = obj_pos[0] - robot_pos[0]
    dy = obj_pos[1] - robot_pos[1]
    norm = math.hypot(dx, dy) or 1.0
    push_dir = (dx/norm, dy/norm)
    return has_push_space(obj_pos, push_dir, known_map)

# ---------- robot pathfinding plan ----------
UNKNOWN, FREE, OCCUPIED = -1, 0, 1
# ---------- add map memory to the system path and import it for several planning ----------
MEMORY_PATH = "map_memory.npy"
if os.path.exists(MEMORY_PATH):
    known_map = np.load(MEMORY_PATH)
    print(f">>> loaded known map from {MEMORY_PATH}")
else:
    known_map = np.full((H, W), UNKNOWN) # robot known map, initial all unknown(updating with exploration)

# -------- real-world map (Only for local perception searching, robot cannot use it for planning directly) ----------
true_map = np.zeros((H, W))
add_solid_to_grid("WALL1", true_map)
add_solid_to_grid("WALL2", true_map)

# ---------------- rasterising forbidden zone ------------
forbidden_mask = np.zeros((H, W))
for r in range(H):
    for c in range(W):
        wx, wy = grid_to_world(c, r)
        cell = Point(wx, wy).buffer(PLAN_RADIUS)   # 半径+安全余量
        # 只要机器人在该格会与禁区相交(非 DC 也非 EC),就禁止进入
        for zone in FORBIDDEN_ZONES:
            rel = rcc8_relation_tol(cell, zone)
            if rel not in ("DC", "EC"):
                forbidden_mask[r, c] = 1
                known_map[r, c] = OCCUPIED        # 对 A* 而言同样不可通行
                break
print(f"Forbidden rasterising finished, occupied {int(forbidden_mask.sum())} grids")

SENSOR_RANGE = 0.3   # perception radius(meter), around 5 grid cells

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

def reset_robot(x, y):
    """move robot back to start position and reset velocity"""
    trans_field = epuck.getField("translation")
    trans_field.setSFVec3f([x, y, 0.0])
    epuck.getField("rotation").setSFRotation([0, 0, 1, 0]) 
    epuck.resetPhysics()      # clear residual speed and inertia
    left.setVelocity(0.0)
    right.setVelocity(0.0)

# ------------------------ multi-trial loop ----------------------------
ep = epuck.getPosition()
START_X, START_Y = ep[0], ep[1]
NUM_TRIALS = 6
all_trajectory = [] # to record all trials' trajectory for plotting comparison
rcc8_log = []

for trial in range(1, NUM_TRIALS + 1):
    print(f"\n>>> Trial {trial} <<<")
    reset_robot(START_X, START_Y)
    for _ in range(5): robot.step(timestep)  # wait a few steps for reset to take effect

    wp_index = 0
    REACH = 0.06        # the distance threshold to reach a waypoint (m)
    GOAL_REACH = 0.08   # the distance threshold to reach the goal
    step_count = 0
    waypoints = []
    replan_count = 0
    trajectory = [] # to record routine that robot has passed
    goal =  None
    goal_found = False
    judged_objects = set()  # to record which movable objects have been judged

    while robot.step(timestep) != -1:
        ep = epuck.getPosition()
        trajectory.append((ep[0], ep[1]))
        step_count += 1

        # ---- perception ----
        found_new = sense(ep[0], ep[1], known_map, true_map)
        # ---- judge whether the movable object is pushable ----
        mbox = robot.getFromDef("MOVABLE_BOX")
        if mbox and "MOVABLE_BOX" not in judged_objects:  # only judge once for each movable object
            mb_pos = mbox.getPosition()
            dist = math.hypot(mb_pos[0]-ep[0], mb_pos[1]-ep[1])
            if dist <= SENSOR_RANGE:                   
                c, r = world_to_grid(mb_pos[0], mb_pos[1])
                if is_pushable("MOVABLE_BOX", ep, known_map):
                    known_map[r, c] = FREE             # pushable → can pass
                    print(f">>> MOVABLE_BOX is pushable, planning to pass through")
                else:
                    hx = 0.05 + ROBOT_RADIUS      # 半边长 + 机器人半径
                    hy = 0.05 + ROBOT_RADIUS
                    c_min, r_min = world_to_grid(mb_pos[0]-hx, mb_pos[1]-hy)
                    c_max, r_max = world_to_grid(mb_pos[0]+hx, mb_pos[1]+hy)
                    for rr in range(max(0,r_min), min(H,r_max+1)):
                        for cc in range(max(0,c_min), min(W,c_max+1)):
                            known_map[rr, cc] = OCCUPIED         # not pushable → treat as obstacle
                    print(f">>> MOVABLE_BOX is not pushable, pass around")
                judged_objects.add("MOVABLE_BOX")  # mark as judged
                # waypoints = []          # empty the old exploration path, force replanning towards the target

        # ---- monitor RCC8 relation in real-time ----
        robot_fp = Point(ep[0], ep[1]).buffer(ROBOT_RADIUS)

        targets = {"DOOR": REGIONS["DOOR"], "FORBIDDEN1": FORBIDDEN_ZONES[0], "FORBIDDEN2": FORBIDDEN_ZONES[1]}
        mbox = robot.getFromDef("MOVABLE_BOX")
        if mbox:
            targets["MOVABLE_BOX"] = make_box_region(mbox.getPosition())

        changes = monitor_rcc8(robot_fp, targets)
        for name, prev, rel in changes:
            print(f"[RCC8] robot vs {name}: {prev} -> {rel}  @step {step_count}")
            rcc8_log.append((step_count, name, prev, rel))   # record for plotting

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
                np.save(MEMORY_PATH, known_map) # save known map to memory
                print(f">>> Arrived at goal.")
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

    all_trajectory.append(list(trajectory)) # record this trial's trajectory
    path_len = sum(math.hypot(trajectory[i+1][0]-trajectory[i][0],
                            trajectory[i+1][1]-trajectory[i][1])
            for i in range(len(trajectory)-1))
    with open("trial_log.txt", "a") as f:
        f.write(f"path length: {path_len:.3f}\t replan count: {replan_count}\n")
    # print(f">>> Trial {trial}: path {path_len:.2f} m, replanned {replan_count}, memory saved to {MEMORY_PATH}.")

# ---------- plot graph about RCC8 relationships changes with time ----------
if rcc8_log:
    # define a "closeness" order for RCC8 relations: DC is farthest, PO/containment is closest
    REL_ORDER = ["DC", "EC", "PO", "TPP", "NTPP", "TPPi", "NTPPi", "EQ"]
    rel_to_y = {rel: i for i, rel in enumerate(REL_ORDER)}

    fig, ax = plt.subplots(figsize=(10, 5))

    # group by object
    objects = sorted(set(name for _, name, _, _ in rcc8_log))
    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))

    for obj, color in zip(objects, colors):
        # take out the relationship change points for this object
        steps = [step for step, name, _, rel in rcc8_log if name == obj]
        rels  = [rel  for step, name, _, rel in rcc8_log if name == obj]
        ys = [rel_to_y[r] for r in rels]
        # steps graph: relationships remain constant between change points
        ax.step(steps, ys, where="post", marker="o",
                color=color, linewidth=2, label=obj)

    ax.set_yticks(range(len(REL_ORDER)))
    ax.set_yticklabels(REL_ORDER)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("RCC8 relation")
    ax.set_title("Evolution of RCC8 relations over time")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.savefig("rcc8_evolution.png", dpi=130, bbox_inches="tight")
    print(">>> exported rcc8_evolution.png")

# ---------- export grid PNG ----------
fig, ax = plt.subplots(figsize=(7, 7))
cmap = matplotlib.colors.ListedColormap(["lightgray", "white", "dimgray"]) # three colors for unknown, free, occupied
ax.imshow(known_map + 1, origin="lower", cmap=cmap, vmin=0, vmax=2)
# robot trajectory(solid line)
colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectory)))
for i, traj in enumerate(all_trajectory):
    g = [world_to_grid(p[0], p[1]) for p in traj]
    ax.plot([p[0] for p in g], [p[1] for p in g],
            "-", color=colors[i], linewidth=2, label=f"Trial {i+1}")
# grid line for each box
ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
ax.grid(which="minor", color="lightgray", linewidth=0.4)

# marked main scaler per 5 boxes
ax.set_xticks(np.arange(0, W, 5))
ax.set_yticks(np.arange(0, H, 5))

rb = red_box.getPosition()
sc, sr = world_to_grid(trajectory[0][0], trajectory[0][1])  # start point
gc, gr = world_to_grid(rb[0], rb[1])
ax.plot(sc, sr, "go", markersize=10, label="Start")
ax.plot(gc, gr, "rs", markersize=10, label="Goal")
# forbidden zone (solid line)
for i, zone in enumerate(FORBIDDEN_ZONES):
    zone_world = list(zone.exterior.coords)
    zone_grid = [world_to_grid(x, y) for x, y in zone_world]
    ax.add_patch(MplPolygon(zone_grid, closed=True,
                            facecolor="red", alpha=0.15,
                            edgecolor="red", linewidth=2.5,
                            label="Forbidden zone (RCC8)" if i == 0 else None))
for i, zone in enumerate(FORBIDDEN_ZONES):
    infl = list(zone.buffer(ROBOT_RADIUS).exterior.coords)
    infl_grid = [world_to_grid(x, y) for x, y in infl]
    ax.add_patch(MplPolygon(infl_grid, closed=True,
                            facecolor="none",
                            edgecolor="red", linewidth=1.2,
                            linestyle=":",
                            label="Zone inflated by robot radius" if i == 0 else None))
def flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

handles, labels = ax.get_legend_handles_labels()
ncol = 2
ax.legend(flip(handles, ncol), flip(labels, ncol), loc="upper left",
          fontsize=8, ncol=ncol, columnspacing=1.0, handletextpad=0.5,
          handlelength=1.5, labelspacing=0.4)
ax.set_title("Robot Trajectories in Multi-Trial Exploration")
plt.savefig("robot_multi_trial_with_forbidden_zones.png", dpi=130, bbox_inches="tight")
print(">>> exported robot_multi_trial_with_forbidden_zones.png")
