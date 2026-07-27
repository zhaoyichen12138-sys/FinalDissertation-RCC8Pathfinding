"""
bfs_pathfinding.py
与 navigator.py 里的 astar() 结构对齐的 BFS 实现，方便直接替换对比。

用法：把主脚本里的
    path = astar(grid, start, goal)
换成
    path = bfs(grid, start, goal)
其余逻辑（航点转换、差速轮控制）完全不用改。
"""

from collections import deque
import math
import time


def bfs(grid, start, goal, return_stats=False):
    """
    grid: H×W 的 numpy 数组，0=可走 1=障碍
    start/goal: (col, row)
    return_stats: True 时额外返回一个 dict，包含 expanded（实际弹出队列处理过的节点数）

    与 astar() 的关键区别：
      - 用 deque 做 FIFO 队列，而不是按 f 值排序的堆
      - 没有启发函数 h，也不比较代价 g，只看"是否访问过"
      - 8 邻域下，扩展顺序本身不区分直走/斜走的真实距离，
        BFS 只保证"跳数"（hop count）最少，不保证真实路径长度最短
      - 没有方向引导，会向四周均匀扩散，expanded 数通常明显高于 A*，
        即使最终路径长度和 A* 一样，也不代表搜索过程一样高效
    """
    H, W = grid.shape
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    visited = {start}
    came = {}
    queue = deque([start])
    expanded = 0   # 真正被弹出队列、处理过邻居的节点数——搜索效率看这个，不要看路径长度

    while queue:
        cur = queue.popleft()          # FIFO：先进队的先处理
        expanded += 1
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
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] == 1:
                continue
            nxt = (nx, ny)
            if nxt in visited:          # 没有 g[] 代价比较，只要没访问过就入队
                continue
            visited.add(nxt)
            came[nxt] = cur
            queue.append(nxt)

    if return_stats:
        return None, {"expanded nodes": expanded}
    return None  # 无路


def path_length_meters(waypoints):
    """给路径算真实欧氏距离总长（米），用于和 A* 做公平对比。
    不要只比较 len(path) 的节点数——8邻域下 BFS 的"跳数最少"
    不等于"真实距离最短"，必须用这个函数换算成米再比较。"""
    total = 0.0
    for i in range(1, len(waypoints)):
        x1, y1 = waypoints[i - 1]
        x2, y2 = waypoints[i]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def run_comparison(astar_fn, grid, start, goal, grid_to_world):
    """
    小工具函数：同一张 grid、同一组 start/goal，
    分别跑一次 A* 和 BFS，打印用于报告的对比数据：
    - 计算耗时（秒）
    - expanded：搜索过程中真正弹出队列处理过的节点数（这才是效率指标，不是路径长度！）
    - path_nodes：最终路径的航点数（可能两者相同，不代表搜索过程一样高效）
    - 路径的真实长度（米）

    注意：astar_fn 需要支持 return_stats=True 参数并返回 (path, {"expanded nodes": N})，
    和本文件里的 bfs() 保持同样的接口。给你自己的 astar() 加同样的 expanded 计数器
    （在 heapq.heappop 之后 +1 即可），否则这里默认对它显示 expanded="N/A"。
    """
    import inspect
    results = {}
    for name, fn in [("A*", astar_fn), ("BFS", bfs)]:
        t0 = time.perf_counter()
        supports_stats = "return_stats" in inspect.signature(fn).parameters
        if supports_stats:
            path, stats = fn(grid, start, goal, return_stats=True)
            expanded = stats.get("expanded nodes", "N/A")
        else:
            path = fn(grid, start, goal)
            expanded = "N/A（need add expanded counter to astar() to display）"
        elapsed = time.perf_counter() - t0

        if path is None:
            results[name] = {"path": None, "time": elapsed, "expanded": expanded}
            continue

        waypoints = [grid_to_world(c, r) for (c, r) in path]
        results[name] = {
            "path": path,
            "waypoints": waypoints,
            "time": elapsed,
            "expanded": expanded,
            "path_nodes": len(path),
            "length_m": path_length_meters(waypoints),
        }

    for name, r in results.items():
        if r["path"] is None:
            print(f"[{name}] did not find the path，costs {r['time']*1000:.2f} ms")
        else:
            print(f"[{name}] costs {r['time']*1000:.2f} ms | "
                  f"expanded(actual search nodes) {r['expanded']} | "
                  f"path nodes {r['path_nodes']} | real length {r['length_m']:.3f} m")

    return results