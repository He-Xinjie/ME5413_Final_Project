# Group5 — Usage

## 1. Install

Copy the `me5413_team_solution/` directory into your catkin workspace:

```bash
cp -r me5413_team_solution ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Dependencies (one-time):

```bash
sudo apt install ros-noetic-move-base ros-noetic-amcl \
                 ros-noetic-map-server ros-noetic-teb-local-planner \
                 ros-noetic-global-planner ros-noetic-costmap-converter \
                 ros-noetic-dynamic-reconfigure
pip3 install ultralytics opencv-python numpy
```

## 2. Launch

Open two terminals.

**Terminal 1 — start the simulation world** (use your usual command to bring
up Gazebo + Jackal + the world).

**Terminal 2 — start navigation + detection + mission orchestrator:**

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch me5413_team_solution navigation.launch
```

This brings up automatically:
- Static map, AMCL, move_base (Dijkstra + TEB)
- RGB-D box detection node (YOLO + 3D dedup)
- Mission orchestrator (listens to RViz Panel commands)
- RViz

## 3. Run the Mission

Click these buttons in the RViz Panel in order:

| Button | Action |
|---|---|
| **Lower Level** | Patrol the lower level, detect and count all boxes |
| **Stairway** | Remove the barrier and traverse the ramp to the upper level |
| **Upper Level** | Scan each upper room, find the target room and enter |
| **Stop** | Immediately cancel the current task |

Recommended order: `Lower -> Stairway -> Upper`. The robot will then
automatically detect, climb, and enter the correct room.

## 4. Customize Waypoints

To re-record waypoints, edit:

```
me5413_team_solution/config/waypoints_lower.yaml      # lower-level patrol path
me5413_team_solution/config/waypoints_stairway.yaml   # ramp traversal path
me5413_team_solution/config/waypoints_upper.yaml      # 4 entrances + 4 in-room targets
```

YAML format:
```yaml
- name: wp_0
  x: 4.10
  y: -0.58
  yaw: 0.0
```

`waypoints_upper.yaml` must contain 8 points: the first 4 are entrance scan
points, the last 4 are in-room target points. They are paired by y-proximity
(wp_0<->wp_5, wp_1<->wp_6, wp_2<->wp_7, wp_3<->wp_4).

## 5. Inspect Results

- **Colored cubes + digit labels in RViz**: every detected box, kept on
  screen after the mission ends so you can verify manually.
- **Terminal logs**: each waypoint arrival, the count summary, the target
  digit list, and the room finally entered.
- **`/box_detector/image`**: annotated camera view (display via an Image
  panel in RViz).

## 6. Algorithm Details

See `me5413_team_solution/docs/SOLUTION.md`.
