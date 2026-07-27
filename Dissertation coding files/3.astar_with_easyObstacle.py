"""navigator: A* 规划 + e-puck with easy obstacle avoidance"""
from controller import Supervisor
import numpy as np
import math
import heapq

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

# ---------- 取机器人朝向(绕竖直 z 轴的偏航角) ----------
def get_heading():
    o = epuck.getOrientation()   # 3x3 旋转矩阵,行主序 9 个数
    # 机器人前方在世界坐标的投影,取 x-y 平面上的角度
    return math.atan2(o[3], o[0])

# ---------- 规划一次 ----------
grid = np.zeros((H, W))          # 暂时全空地,后面再加障碍
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
print(f"障碍中心({ob_pos[0]:.2f},{ob_pos[1]:.2f}) 尺寸({ob_size[0]:.2f},{ob_size[1]:.2f})")
print(f"障碍占据栅格 col[{c_min}~{c_max}] row[{r_min}~{r_max}]")

ep = epuck.getPosition()
rb = red_box.getPosition()
start = world_to_grid(ep[0], ep[1])
goal = world_to_grid(rb[0], rb[1])
path = astar(grid, start, goal)

if path is None:
    print("找不到路径")
    waypoints = []
else:
    # 把栅格路径转成世界坐标航点
    waypoints = [grid_to_world(c, r) for (c, r) in path]
    print(f"路径共 {len(waypoints)} 个航点")

wp_index = 0
REACH = 0.06        # 到达航点的距离阈值(米)
GOAL_REACH = 0.08   # 到达终点阈值

while robot.step(timestep) != -1:
    if not waypoints or wp_index >= len(waypoints):
        left.setVelocity(0.0)
        right.setVelocity(0.0)
        continue

    ep = epuck.getPosition()
    tx, ty = waypoints[wp_index]
    dx, dy = tx - ep[0], ty - ep[1]
    dist = math.hypot(dx, dy)

    # 到达当前航点,切下一个
    if dist < REACH:
        wp_index += 1
        if wp_index >= len(waypoints):
            print(">>> 已到达红盒子附近")
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
