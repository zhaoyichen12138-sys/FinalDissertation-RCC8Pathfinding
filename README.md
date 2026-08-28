# Final Dissertation — Robot Path-finding and Object Searching in a Grid-based Environment with RCC8

This repository is the complete code documentation for the final dissertation of **Yichen Zhao**, which investigates how robot pathfinding in open areas can be combined with **RCC8** qualitative spatial reasoning.

It contains **all of the code files** used in the project, together with the supporting materials produced during the research: experimental data logs, generated figures, saved map memories, and simulation videos.

---

## 1. Repository Overview

```
Final_Dissertation-RCC8-pathfinding-/
├── Dissertation coding files/     # Standalone scripts, in the order they were developed
├── Webots_project/                # The complete Webots simulation project
│   ├── worlds/                    # Scene (.wbt) files — open these in Webots
│   ├── controllers/               # Robot behaviour code (one folder per controller)
│   ├── videos/                    # Recorded simulation videos
│   ├── protos/ libraries/ plugins/# Webots project folders (standard, mostly unused)
└── README.md
```

### 1.1 `Dissertation coding files/`

The numbered scripts trace the progression of the dissertation, from a plain grid search to the final RCC8-informed planner:

| File | Content |
| --- | --- |
| `1.bfs_pathfindingAndComparison.py` | Baseline BFS pathfinding and comparison utilities |
| `2.astar_base_navigator.py` | Basic A* navigator with e-puck differential drive control |
| `3.navigator_withRCC8(without_obstacle).py` | A* combined with RCC8 relations, no physical obstacles |
| `4.mustPassDoor_RCC8_relation.py` | Topological-layer reasoning (must-pass-door) before metric A* planning |
| `5.localPerception.py` | Planning from locally perceived information instead of a global map |
| `6.localPerception_withMulti_trials.py` | Local perception with map memory across repeated trials |
| `7.detect_movable_object.py` | Detecting and reasoning about movable objects |
| `8.A*_penaltyFunction.py` | A* with a penalty function: push the obstacle, or detour around it |

### 1.2 `Webots_project/controllers/`

Each folder is one controller — the code that decides how the robot behaves in simulation:

| Controller | Behaviour |
| --- | --- |
| `navigator` | A* planning with a BFS comparison, driving the e-puck to the red box |
| `navigator_withRCC8(without_obstacle)` | A* constrained by RCC8 relations / forbidden zones |
| `mustPassDoor` | Plans spatial relations on the topological layer first, then A* on the metric layer |
| `localPerception_withMemory` | Builds and updates a remembered map from local perception, replanning on the known map |
| `detect_movable_object` | Identifies movable objects and tracks how the RCC8 relations evolve |
| `Astar_with_Penalty` | Penalty-based decision between pushing a movable object and detouring |
| `diagnostic` | Helper controller that verifies the world-to-grid coordinate conversion |

---

## 2. Running the Simulation in Webots

### 2.1 Install Webots

Webots is a free, open-source robot simulator.

- **Download:** https://cyberbotics.com/#download
- **Official documentation:** https://cyberbotics.com/doc/guide/index

Install the version matching your operating system (Windows / macOS / Linux) and follow the official installation guide.

The controllers are written in Python and rely on `numpy`, `matplotlib` and `shapely`, so make sure these packages are available to the Python interpreter that Webots uses.

### 2.2 Open a scene

After launching Webots, simply **open a world file from the `Webots_project/worlds/` folder** directly:

> `File → Open World…` → choose e.g. `Webots_project/worlds/Robot_pathfinding_RCC8.wbt`

No further project configuration is needed — the scene loads with its walls, target box and robot already in place.

### 2.3 Switch the controller

The `controllers/` folder holds the files that control the robot's behaviour. To run a different experiment on the same scene:

1. In the simulation window (or the scene tree), **select the `e-puck` robot**.
2. Open its **properties / field list** and find the `controller` field.
3. **Change the `controller` value** to the name of the controller folder you want, e.g. `navigator`, `mustPassDoor`, or `Astar_with_Penalty`.
4. Reload the world and press play — the robot now runs the selected behaviour.

Each run writes its logs and figures back into that controller's own folder, so results from different behaviours stay separated.

---

## 3. Simulation Videos

The behaviours that have already been implemented and run were recorded, and the videos are kept in **`Webots_project/videos/`** for direct viewing without needing to re-run the simulation:

| Video | Shows |
| --- | --- |
| `basic_pathfinding.mp4` | Baseline pathfinding to the target |
| `mustPassDoor.mp4` | Topological must-pass-door reasoning followed by metric planning |
| `AutoPerception_with_MultiTrials.mp4` | Local perception with map memory over repeated trials |
| `Push_with_LowPenalty.mp4` | With a low penalty, the robot pushes the movable object through |
| `Detour_with_HighPenalty.mp4` | With a high penalty, the robot detours around it instead |

The last two are the same scene under different penalty settings, and are best watched together to see the effect of the penalty function.
