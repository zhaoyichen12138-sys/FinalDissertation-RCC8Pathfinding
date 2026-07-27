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

# ---------- A* ----------
def astar(grid, start, goal, return_stats=False):
    # grid: H×W, 0=可走 1=障碍; start/goal 是 (col,row)
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
    return None   # 无路

# ---------- 取机器人朝向(绕竖直 z 轴的偏航角) ----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 旋转矩阵,行主序 9 个数
    # 机器人前方在世界坐标的投影,取 x-y 平面上的角度
    return math.atan2(o[3], o[0])

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

ROBOT_RADIUS = 0.035   # e-puck 半径约 3.5cm

def robot_footprint(x, y):
    """把机器人近似成一个圆形区域"""
    return Point(x, y).buffer(ROBOT_RADIUS)

# ---------- 规划一次 ----------
grid = np.zeros((H, W))          # 暂时全空地,后面再加障碍

# ---------------- rasterising forbidden zone ------------
forbidden_mask = np.zeros((H, W))

for r in range(H):
    for c in range(W):
        wx, wy = grid_to_world(c, r)
        cell = Point(wx, wy).buffer(ROBOT_RADIUS)
        # 只要机器人在该格会与禁区相交(非 DC 也非 EC),就禁止进入
        for zone in FORBIDDEN_ZONES:
            rel = rcc8_relation(cell, zone)
            if rel not in ("DC", "EC"):
                forbidden_mask[r, c] = 1
                grid[r, c] = 1        # 对 A* 而言同样不可通行
                break
print(f"Forbidden rasterising finished, occupied {int(forbidden_mask.sum())} grids")

ep = epuck.getPosition()
rb = red_box.getPosition()
start = world_to_grid(ep[0], ep[1])
goal = world_to_grid(rb[0], rb[1])

run_comparison(astar, grid, start, goal, grid_to_world)
path = astar(grid, start, goal)
#path = bfs(grid, start, goal)

if path is None:
    print("Cannot find the path")
    waypoints = []
else:
    # 把栅格路径转成世界坐标航点
    waypoints = [grid_to_world(c, r) for (c, r) in path]
    print(f"Totally {len(waypoints)} checkpoints through the path")

wp_index = 0
REACH = 0.06        # 到达航点的距离阈值(米)
GOAL_REACH = 0.08   # 到达终点阈值

trajectory = [] # to record routine that robot has passed

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
            print(f"!!! Violate constraint:relation with zone {i+1} is {rel}")
    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    dist = math.hypot(dx, dy)

    # 到达当前航点,切下一个
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
                                        label="Forbidden zone (RCC8)" if 1 == 0 else None))

            # 2. A* 规划的路径(虚线)
            wp_x = [p[0] for p in waypoints]
            wp_y = [p[1] for p in waypoints]
            ax.plot(wp_x, wp_y, "b--", linewidth=1.5,
                    marker="o", markersize=4, label="A* planned path")

            # 3. 机器人实际轨迹(实线)
            tr_x = [p[0] for p in trajectory]
            tr_y = [p[1] for p in trajectory]
            ax.plot(tr_x, tr_y, "g-", linewidth=2, label="Actual trajectory")

            # 4. start and stop points
            ax.plot(tr_x[0], tr_y[0], "go", markersize=12, label="Start")
            ax.plot(rb[0], rb[1], "rs", markersize=12, label="Target (red box)")

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Constraint-aware path: robot avoids forbidden zone")
            ax.legend(loc="upper left", fontsize=9)

            plt.savefig("avoids_forbiddenZone_path.png", dpi=150, bbox_inches="tight")
            print(">>> exported avoids_forbiddenZone_path.png")
        continue

    # 计算需要的转向:目标方向 - 当前朝向
    target_angle = math.atan2(dy, dx)
    heading = get_heading()
    err = target_angle - heading
    # 归一化到 [-pi, pi]
    err = math.atan2(math.sin(err), math.cos(err))

    # 简单控制:朝向偏差大就原地转,偏差小就直行
    if abs(err) > 0.3:
        #print(f"转向中 heading={heading:.2f} target={target_angle:.2f} err={err:.2f}")
        turn = 2.0 if err > 0 else -2.0
        left.setVelocity(-turn)
        right.setVelocity(turn)
    else:
        # 直行 + 轻微修正
        base = 0.5 * MAX_SPEED
        corr = 2.0 * err
        left.setVelocity(base - corr)
        right.setVelocity(base + corr)
