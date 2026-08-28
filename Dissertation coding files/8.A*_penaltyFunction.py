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

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")
movable_box = robot.getFromDef("MOVABLE_BOX")

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
    if a.intersects(b):
        if a.equals(b): return "EQ"
        if a.within(b): return "TPP" if a.boundary.intersects(b.boundary) else "NTPP"
        if b.within(a): return "TPPi" if a.boundary.intersects(b.boundary) else "NTPPi"
        if a.touches(b): return "EC"
        return "PO"
    if a.distance(b) <= tol:
        return "EC"          
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
ROBOT_RADIUS = 0.035   
#SAFETY_MARGIN = 0.03
PLAN_RADIUS = ROBOT_RADIUS

def robot_footprint(x, y):
    """similar to the function for creating a circle region"""
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
def astar(grid, start, goal, cost_map):
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
            extra = cost_map[ny, nx] if cost_map is not None else 0  # add extra cost for pushing movable object
            ng = g[cur] + step + extra
            nxt = (nx, ny)
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_set, (ng + h(nxt, goal), nxt))
    return None

# ---------- obtain robot heading direction----------
def get_heading():
    o = epuck.getOrientation()   
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
    if not (0 <= c < W and 0 <= r < H): #judge by DC relation
        return False                    # out of grid, no space to push
        print(f">>> probe point ({probe_x:.2f}, {probe_y:.2f}) is out of grid, cannot push")
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

def planning_map(known_map, forbidden_mask=None, temp_obstacle=None):
    """treat as accessble area if unknown, and do A* planning"""
    m = known_map.copy()
    m[m == UNKNOWN] = FREE
    if forbidden_mask is not None:
        m[forbidden_mask == 1] = OCCUPIED
    if temp_obstacle is not None:
        m[temp_obstacle == 1] = OCCUPIED
    return m

PUSH_PENALTY = 4 # extra cost for pushing a movable object(means more 15 grid cells)
cost_map = np.zeros((H, W)) # cost map for A* planning, default 0 for free space

# ---------- robot pathfinding plan ----------
UNKNOWN, FREE, OCCUPIED = -1, 0, 1
# ---------- add map memory to the system path and import it for several planning ----------
MEMORY_PATH = "map_memory.npz"
if os.path.exists(MEMORY_PATH):
    data = np.load(MEMORY_PATH)
    known_map = data["known_map"]
    saved_goal = tuple(int(x) for x in data["goal"])
    print(f">>> loaded known map from {MEMORY_PATH}")
else:
    known_map = np.full((H, W), UNKNOWN) # robot known map, initial all unknown(updating with exploration)
    saved_goal = (-1, -1)

# -------- real-world map (Only for local perception searching, robot cannot use it for planning directly) ----------
true_map = np.zeros((H, W))
add_solid_to_grid("WALL1", true_map)
add_solid_to_grid("WALL2", true_map)
add_solid_to_grid("WALL3", true_map)

# ---------------- rasterising forbidden zone ------------
forbidden_mask = np.zeros((H, W))
for r in range(H):
    for c in range(W):
        wx, wy = grid_to_world(c, r)
        cell = Point(wx, wy).buffer(PLAN_RADIUS)   
        for zone in FORBIDDEN_ZONES:
            rel = rcc8_relation_tol(cell, zone)
            if rel not in ("DC", "EC"):
                forbidden_mask[r, c] = 1
                known_map[r, c] = OCCUPIED        
                break
print(f"Forbidden rasterising finished")

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
rb = red_box.getPosition()
mbox_init = movable_box.getPosition()[:2] #initial position of the movable box, because the position will change if door is pushed by robot
START_X, START_Y = ep[0], ep[1]
NUM_TRIALS = 1
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
    judged_objects = set()  # to record which movable objects have been judged
    goal =  None
    goal_found = False
    if saved_goal != (-1, -1):
        goal = saved_goal
        goal_found = True
        print(f">>> recalled goal from memory: {goal}")

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
                    hx = 0.05 + ROBOT_RADIUS      
                    hy = 0.05 + ROBOT_RADIUS
                    c_min, r_min = world_to_grid(mb_pos[0]-hx, mb_pos[1]-hy)
                    c_max, r_max = world_to_grid(mb_pos[0]+hx, mb_pos[1]+hy)
                    for rr in range(max(0,r_min), min(H,r_max+1)):
                        for cc in range(max(0,c_min), min(W,c_max+1)):
                            cost_map[rr, cc] = PUSH_PENALTY  # add extra cost for pushing
                    print(f">>> MOVABLE_BOX is pushable (penalty={PUSH_PENALTY})")
                else:
                    hx = 0.05 + ROBOT_RADIUS      
                    hy = 0.05 + ROBOT_RADIUS
                    c_min, r_min = world_to_grid(mb_pos[0]-hx, mb_pos[1]-hy)
                    c_max, r_max = world_to_grid(mb_pos[0]+hx, mb_pos[1]+hy)
                    for rr in range(max(0,r_min), min(H,r_max+1)):
                        for cc in range(max(0,c_min), min(W,c_max+1)):
                            temp_obstacle[rr, cc] = 1 
                    print(f">>> MOVABLE_BOX is not pushable, pass around")
                judged_objects.add("MOVABLE_BOX")  # mark as judged
                # print(f"path length (cells) = {len(new_path)}")
                # passes_box = any(cost_map[r, c] > 0 for (c, r) in new_path)
                # print(f"path passes through box (push it?) {passes_box}")
                waypoints = []          # empty the old exploration path, force replanning towards the target
        # ---- monitor RCC8 relation in real-time ----
        robot_fp = Point(ep[0], ep[1]).buffer(ROBOT_RADIUS)

        targets = {"DOOR": REGIONS["DOOR"], "FORBIDDEN1": FORBIDDEN_ZONES[0], "FORBIDDEN2": FORBIDDEN_ZONES[1]}
        mbox = robot.getFromDef("MOVABLE_BOX")
        if mbox:
            targets["MOVABLE_BOX"] = make_box_region(mbox.getPosition())

        changes = monitor_rcc8(robot_fp, targets)
        for name, prev, rel in changes:
            # print(f"[RCC8] robot vs {name}: {prev} -> {rel}  @step {step_count}")
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

            new_path = astar(pmap, start, target_cell, cost_map)
            if new_path:
                waypoints = [grid_to_world(c, r) for (c, r) in new_path]
                wp_index = 1 if len(waypoints) > 1 else 0
                replan_count += 1
                # --- penalty decision statistics ---
                if "MOVABLE_BOX" in judged_objects and goal_found:
                    n = sum(1 for (c, r) in new_path if cost_map[r, c] > 0)
                    decision = "PUSH" if n > 0 else "DETOUR"
                    L = sum(math.hypot(new_path[i+1][0]-new_path[i][0],new_path[i+1][1]-new_path[i][1])
                            for i in range(len(new_path)-1))
                    print(f"[penalty={PUSH_PENALTY}] cells={len(new_path)},"
                          f"geometric length={L:.2f}, n={n}, decision={decision}")

        # ---- judge whether goal is reached ----
        if goal_found:
            rb = red_box.getPosition()
            dist_to_goal = math.hypot(rb[0] - ep[0], rb[1] - ep[1])
            if dist_to_goal < GOAL_REACH:
                left.setVelocity(0.0)
                right.setVelocity(0.0)
                np.savez(MEMORY_PATH, known_map=known_map, goal=np.array(goal if goal else [-1, -1])) # save known map to memory
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
    print(f">>> Trial {trial}: path {path_len:.2f} m, replanned {replan_count}, memory saved to {MEMORY_PATH}.")

# ---- Plan twice with different penalties for comparison ----
def plan_with_penalty(P):
    cm = np.zeros((H, W))
    for rr in range(max(0, r_min), min(H, r_max + 1)):
        for cc in range(max(0, c_min), min(W, c_max + 1)):
            cm[rr, cc] = P
    p = astar(planning_map(known_map), start, goal, cm)
    L = sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1])
            for i in range(len(p)-1))
    n = sum(1 for (c, r) in p if cm[r, c] > 0)
    return p, L, n

path_lo, L_lo, n_lo = plan_with_penalty(3.0)
path_hi, L_hi, n_hi = plan_with_penalty(3.5)
print(f"P=3.0: length={L_lo:.2f}, n={n_lo}")
print(f"P=3.5: length={L_hi:.2f}, n={n_hi}")

# ---------- export grid PNG ----------
fig, ax = plt.subplots(figsize=(7, 7))
cmap = matplotlib.colors.ListedColormap(["lightgray", "white", "dimgray"]) # three colors for unknown, free, occupied
ax.imshow(known_map + 1, origin="lower", cmap=cmap, vmin=0, vmax=2)
# robot trajectory(solid line)
colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectory)))
for p, col, ls, lab in [
        (path_lo, "#1f4e79", "-",  f"$Penalty = 3.0$  (push, $L$ = {L_lo:.1f})"),
        (path_hi, "#c00000", "--", f"$Penalty = 3.5$  (detour, $L$ = {L_hi:.1f})")]:
    ax.plot([c for (c, r) in p], [r for (c, r) in p],
            ls, color=col, linewidth=2, label=lab)
# grid line for each box
ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
ax.grid(which="minor", color="lightgray", linewidth=0.4)

# marked main scaler per 5 boxes
ax.set_xticks(np.arange(0, W, 5))
ax.set_yticks(np.arange(0, H, 5))
# mark the start and goal points, and the door's region
sc, sr = world_to_grid(trajectory[0][0], trajectory[0][1])  # start point
gc, gr = world_to_grid(rb[0], rb[1])
mb_half = 0.05 + ROBOT_RADIUS
mb_corners_world = [
    (mbox_init[0]-mb_half, mbox_init[1]-mb_half),
    (mbox_init[0]+mb_half, mbox_init[1]-mb_half),
    (mbox_init[0]+mb_half, mbox_init[1]+mb_half),
    (mbox_init[0]-mb_half, mbox_init[1]+mb_half),
]
mb_corners_grid = [world_to_grid(x, y) for x, y in mb_corners_world]
ax.add_patch(MplPolygon(mb_corners_grid, closed=True,
                        facecolor="lime", alpha=0.35,
                        edgecolor="green", linewidth=2,
                        label="Door"))
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
                            
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Decision making with Penalty")
plt.savefig("robot_astar_with_penalty.png", dpi=130, bbox_inches="tight")
print(">>> exported robot_astar_with_penalty.png")
