"""diagnostic: check World to Grid conversion and export comparison image"""
from controller import Supervisor
import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck = robot.getFromDef("EPUCK")
red_box = robot.getFromDef("RED_BOX")

# ---------- Grid parameters ----------
RES = 0.05                 # 5cm per cell
ORIGIN = (-1.0, -1.0)      # Grid [0,0] corresponds to world coordinates (x, y)
W, H = 40, 40              # 2m / 0.05 = 40 cells, covering x,y ∈ [-1, 1]

def world_to_grid(wx, wy):
    col = int((wx - ORIGIN[0]) / RES)
    row = int((wy - ORIGIN[1]) / RES)
    return col, row

def grid_to_world(col, row):
    wx = ORIGIN[0] + (col + 0.5) * RES
    wy = ORIGIN[1] + (row + 0.5) * RES
    return wx, wy

# ---------- self checking: go and back ----------
wx, wy = 0.37, -0.12
c, r = world_to_grid(wx, wy)
bx, by = grid_to_world(c, r)
print(f"tolerance(meter): {abs(wx-bx):.4f}, {abs(wy-by):.4f}  (should less than {RES})")

exported = False
while robot.step(timestep) != -1:
    if not exported:
        ep = epuck.getPosition()
        rb = red_box.getPosition()

        # use the ground plane's first two axes (x, y)
        ep_c, ep_r = world_to_grid(ep[0], ep[1])
        rb_c, rb_r = world_to_grid(rb[0], rb[1])
        print(f"e-puck world({ep[0]:.2f},{ep[1]:.2f}) -> grid({ep_c},{ep_r})")
        print(f"red_box world({rb[0]:.2f},{rb[1]:.2f}) -> grid({rb_c},{rb_r})")

        # draw grid:0=empty, 1=robot, 2=red box
        grid = np.zeros((H, W))
        if 0 <= ep_r < H and 0 <= ep_c < W:
            grid[ep_r, ep_c] = 1
        if 0 <= rb_r < H and 0 <= rb_c < W:
            grid[rb_r, rb_c] = 2

       # ---------- Plotting: white background + grid lines ----------
        fig, ax = plt.subplots(figsize=(6, 6))

        # white background, only draw cells with content
        cmap = matplotlib.colors.ListedColormap(["white", "tab:blue", "red"])
        ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=2)

        # one grid line per cell: minor ticks on cell boundaries
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.5)

        # mark the number every 5 ticks
        ax.set_xticks(np.arange(0, W, 5))
        ax.set_yticks(np.arange(0, H, 5))

        ax.set_title("grid check: robot=blue, red_box=red")
        ax.set_xlabel("col (x)")
        ax.set_ylabel("row (y)")

        plt.savefig("grid_check.png", dpi=120)
        print(">>> 已导出 grid_check.png")
        exported = True