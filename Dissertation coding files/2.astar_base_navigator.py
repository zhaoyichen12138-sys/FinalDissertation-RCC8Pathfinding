"""navigator: A* planning + e-puck differential drive control, to reach the red box and compare with BFS"""
from controller import Supervisor
from bfs_pathfinding import bfs, run_comparison
import numpy as np
import math
import heapq

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
    # grid: H×W, 0=traversable, 1=obstacle; start/goal is (col,row)
    def h(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0, start)]
    came = {}
    g = {start: 0}
    # 8 邻域
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
        return None, {"expanded nodes", expanded}
    return None   

# ---------- get robot heading ----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 rotation matrix, row-major 9 elements
    # project the robot's front direction onto the world coordinate system, take the angle on the x-y plane
    return math.atan2(o[3], o[0])

# ---------- path planning ----------
grid = np.zeros((H, W))          
#import a base obstacle（wall）
obstacle = robot.getFromDef("WALL")
grid = np.zeros((H,W))
ob_pos = obstacle.getPosition() #read the location and size of 'WALL'
size_field = obstacle.getField("children").getMFNode(0).getField("geometry").getSFNode().getField("size")
ob_size = size_field.getSFVec3f()
#coordinate value(x,y,z) correspond respectively to (0,1,2) in the array 
half_x, half_y = ob_size[0]/2, ob_size[1]/2 
x_min, x_max = ob_pos[0] - half_x, ob_pos[0] + half_x
y_min, y_max = ob_pos[1] - half_y, ob_pos[1] + half_y
c_min, r_min = world_to_grid(x_min, y_min)
c_max, r_max = world_to_grid(x_max, y_max)
for r in range(max(0, r_min), min(H, r_max + 1)):
    for c in range(max(0, c_min), min(W, c_max + 1)):
        grid[r,c] = 1
# print(f"Centre of obstacle({ob_pos[0]:.2f},{ob_pos[1]:.2f}) Size({ob_size[0]:.2f},{ob_size[1]:.2f})")
# print(f"Obstacle occupied the grid col[{c_min}~{c_max}] row[{r_min}~{r_max}]")

ep = epuck.getPosition()
rb = red_box.getPosition()
start = world_to_grid(ep[0], ep[1])
goal = world_to_grid(rb[0], rb[1])

run_comparison(astar, grid, start, goal, grid_to_world)
path = astar(grid, start, goal)

if path is None:
    print("Cannot find the path")
    waypoints = []
else:
    # transform the grid path to world coordinates waypoints
    waypoints = [grid_to_world(c, r) for (c, r) in path]
    print(f"Totally {len(waypoints)} checkpoints through the path")

wp_index = 0
REACH = 0.06        # threshold to reach a waypoint (meters)
GOAL_REACH = 0.08   # threshold to reach the goal (meters)

while robot.step(timestep) != -1:
    if not waypoints or wp_index >= len(waypoints):
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        continue

    ep = epuck.getPosition()
    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    dist = math.hypot(dx, dy)

    # arrive at current waypoint, move to the next
    if dist < REACH:
        wp_index += 1
        if wp_index >= len(waypoints):
            print(">>> Arrived Red_Box nearby")
        continue

    # calculate the required turn: target direction - current heading
    target_angle = math.atan2(dy, dx)
    heading = get_heading()
    err = target_angle - heading
    # normalize to [-pi, pi]
    err = math.atan2(math.sin(err), math.cos(err))

    # easy control: rotate if heading deviation is large, otherwise go straight
    if abs(err) > 0.3:
        #print(f"转向中 heading={heading:.2f} target={target_angle:.2f} err={err:.2f}")
        turn = 2.0 if err > 0 else -2.0
        left.setVelocity(-turn)
        right.setVelocity(turn)
    else:
        # go straight + revise lightly
        base = 0.5 * MAX_SPEED
        corr = 2.0 * err
        left.setVelocity(base - corr)
        right.setVelocity(base + corr)
