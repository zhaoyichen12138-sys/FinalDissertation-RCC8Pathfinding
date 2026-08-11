"""diagnostic: 验证 世界坐标 -> 栅格 的转换,并导出对比图"""
from controller import Supervisor
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无窗口后台绘图,存文件用
import matplotlib.pyplot as plt

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")

# ---------- 栅格参数 ----------
RES = 0.05                 # 每格 5cm
ORIGIN = (-1.0, -1.0)      # 栅格[0,0]对应的世界坐标 (x, y)
W, H = 40, 40              # 2m / 0.05 = 40 格,覆盖 x,y ∈ [-1, 1]

def world_to_grid(wx, wy):
    col = int((wx - ORIGIN[0]) / RES)
    row = int((wy - ORIGIN[1]) / RES)
    return col, row

def grid_to_world(col, row):
    wx = ORIGIN[0] + (col + 0.5) * RES
    wy = ORIGIN[1] + (row + 0.5) * RES
    return wx, wy

# ---------- 往返自检 ----------
wx, wy = 0.37, -0.12
c, r = world_to_grid(wx, wy)
bx, by = grid_to_world(c, r)
print(f"往返误差(米): {abs(wx-bx):.4f}, {abs(wy-by):.4f}  (应小于 {RES})")

exported = False
while robot.step(timestep) != -1:
    if not exported:
        ep = epuck.getPosition()
        rb = red_box.getPosition()

        # 用地面平面的前两个轴 (x, y)
        ep_c, ep_r = world_to_grid(ep[0], ep[1])
        rb_c, rb_r = world_to_grid(rb[0], rb[1])
        print(f"e-puck 世界({ep[0]:.2f},{ep[1]:.2f}) -> 栅格({ep_c},{ep_r})")
        print(f"red_box 世界({rb[0]:.2f},{rb[1]:.2f}) -> 栅格({rb_c},{rb_r})")

        # 画栅格:0=空, 1=机器人, 2=红盒子
        grid = np.zeros((H, W))
        if 0 <= ep_r < H and 0 <= ep_c < W:
            grid[ep_r, ep_c] = 1
        if 0 <= rb_r < H and 0 <= rb_c < W:
            grid[rb_r, rb_c] = 2

        # ---------- 绘图:白底 + 网格线 ----------
        # fig, ax = plt.subplots(figsize=(6, 6))

        # 白底,只画有内容的格子
        # cmap = matplotlib.colors.ListedColormap(["white", "tab:blue", "red"])
        # ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=2)

        # 每格一条网格线:次刻度落在格子边界上
        # ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        # ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        # ax.grid(which="minor", color="lightgray", linewidth=0.5)

        # 主刻度每 5 格标一个数字
        # ax.set_xticks(np.arange(0, W, 5))
        # ax.set_yticks(np.arange(0, H, 5))

        # ax.set_title("grid check: robot=blue, red_box=red")
        # ax.set_xlabel("col (x)")
        # ax.set_ylabel("row (y)")

        # plt.savefig("grid_check.png", dpi=120)
        # print(">>> 已导出 grid_check.png")
        exported = True