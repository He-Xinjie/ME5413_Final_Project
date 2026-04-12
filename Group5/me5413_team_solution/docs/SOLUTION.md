# Solution Documentation

This document describes the overall navigation pipeline of the solution,
covering the three phases (lower / stairway / upper), box detection and
deduplication, counting strategy, and how the mission is finally completed.

---

## 0. System Architecture

```
┌──────────────┐    /front/image_raw      ┌────────────────────┐
│ Simulation   │───/front/depth/image_raw──▶│ box_detector_3d.py │
│ (Jackal+RGB-D)│   /front/scan             └─────────┬──────────┘
└──────┬───────┘                                       │ /box_detector/counts
       │                                               │ /mission_planner/min_box_id (latched)
       │                                               │ /box_detector/markers
       ▼                                               ▼
┌──────────────┐   /move_base                   ┌────────────────┐
│ move_base    │◀───────────────────────────────│ nav_test.py    │
│ (Dijkstra+TEB)│   action goals                 │ (ZoneNavigator)│
└──────────────┘                                 └────────────────┘
                                                   ▲
                                                   │ /nav_zone/command
                                                   │  (lower / stairway / upper / stop)
                                              RViz Panel
```

`nav_test.py` is the mission orchestrator. It runs the lower / stairway /
upper phases according to commands from the RViz Panel.
`box_detector_3d.py` continuously runs box detection, 3D deduplication, and
counting in the background. `move_base` handles the actual navigation using
Dijkstra (global) + TEB (local).

---

## 1. Box Detection and Counting (box_detector_3d.py)

### 1.1 Detection Pipeline

Each synchronized RGB-D message enters `rgbd_callback` and goes through:

1. **YOLO v8 detection**: `models/Final.pt` runs on the RGB image and
   returns bounding boxes plus a digit class (1-9).
2. **Depth sampling**: median depth in a 5x5 pixel window at the bbox center,
   discarding NaN/0.
3. **Pixel -> camera frame**: back-projection using camera intrinsics
   (fx, fy, cx, cy) into a 3D point.
4. **Camera frame -> map frame**: TF transform
   (`map <- base_link <- front_camera_optical`) into global map coordinates.
5. **Height filter**: drop points with `mz < -0.5` or `mz > 4.0` (ground or
   ceiling false positives).
6. **Clustering + voting**: `find_or_create_candidate` either merges with an
   existing candidate or creates a new one.

### 1.2 Deduplication Strategy

Each `BoxCandidate` represents one **physical box** in the global map:

```python
class BoxCandidate:
    x, y, z          # weighted-average map coordinates
    obs_count        # how many times observed
    votes: dict      # digit -> vote count
    last_seen        # last update timestamp
    scan_id          # scan window when first created (used for upper room isolation)
    phase            # 'lower' / 'upper', determines counting attribution
```

**Core rules**:

* `same_digit_radius = 2.0 m`: two same-digit candidates within 2.0 m are
  treated as one physical box.
* `cluster_radius = 2.5 m`: kept for debugging only.
* **Different digits never merge**: avoids merging neighboring boxes with
  different numbers.
* **Costmap gap check** (`_has_gap_between`): even with matching digit and
  close distance, if the straight line between two candidates passes through
  an occupied cell (wall, furniture), they are kept as two separate boxes.
* **Per-candidate cooldown** (`cooldown_seconds = 1.0`): a candidate can be
  voted on at most once per second, preventing burst-voting from consecutive
  frames.
* **Position EMA**: `add_vote` uses `1/obs_count` exponential averaging so a
  single bad depth reading cannot drag the position away.
* **Periodic merge** (`merge_nearby_candidates`): every 5 s a global pass
  merges any pair that "wasn't sure of its digit when first created but
  later became the same digit".

### 1.3 Counting

* `is_confirmed(min_votes=3)`: a candidate is "confirmed" once it has at
  least 3 votes and the top digit holds at least 50% of the vote share.
* `get_deduplicated_counts()`: walks all confirmed candidates with
  **`phase == 'lower'`** and aggregates `{digit: count}` by `best_digit()`.
  **Boxes discovered in the upper phase do NOT enter this count.**
* `get_fresh_counts()`: returns only confirmed candidates whose
  `scan_id == current_scan_id`. Used by the upper phase so each room can
  judge what is currently visible in isolation.

### 1.4 Topics Published

| Topic | Content |
|---|---|
| `/box_detector/counts` | Published every frame. lower phase: `get_deduplicated_counts()`; upper phase: `get_fresh_counts()` |
| `/mission_planner/min_box_id` (latched) | **Only published in lower phase.** Digits with the minimum count, comma-separated when tied (e.g. `"2,8"`). Locked when entering upper, never overwritten |
| `/box_detector/markers` | RViz markers for all confirmed candidates. `lifetime=0` for permanent display, with DELETEALL+ADD each frame to avoid stale ids |
| `/box_detector/image` | Annotated visualization image |

### 1.5 Phase Control

`zone_cmd_callback` listens to `/nav_zone/command`:

| Command | Behavior |
|---|---|
| `lower` | Enable detection, phase='lower' |
| `stairway` / `stop` | Disable detection (the robot's view is distorted on the ramp; turning it off avoids polluted data) |
| `upper` | Enable detection, phase='upper', **bump scan_id** (do NOT clear candidates, all markers preserved); also `min_box_locked=True` |
| `/box_detector/reset` (Empty) | Bump scan_id, do NOT clear candidates. nav_test sends one before each upper room scan |

---

## 2. Lower Phase: Obstacle-Avoidance Navigation

### 2.1 Goal

Patrol the entire lower-level box area following a zigzag path defined in
`config/waypoints_lower.yaml`, so the camera covers every box and
box_detector_3d completes detection and counting.

### 2.2 Algorithm

* **Global planner**: `global_planner/GlobalPlanner` with `use_dijkstra=true`,
  driven by static map + ObstacleLayer + InflationLayer.
* **Local planner**: `teb_local_planner` with Homotopy Class Planning enabled
  (up to 4 trajectory classes; tries multiple paths in parallel in cluttered
  environments).
* **Obstacle source**: a single 2D laser `front/scan`, written into the
  costmap by ObstacleLayer.
* **Inflation**: `inflation_radius=1.0 m`, `cost_scaling_factor=3.0`.

### 2.3 Flow (`run_lower_level`)

1. Load all waypoints from `waypoints_lower.yaml`.
2. For each waypoint:
   1. **`_adjust_goal`**: check the global costmap to see whether the goal
      is occupied by a randomly spawned box (`cost >= 80`). If so, search
      eight directions at radii 0.7 m / 1.2 m / 1.8 m for a free cell
      (`cost < 50`) to use as a substitute, so move_base does not fail
      simply because the target sits on a lethal cell.
   2. **`send_goal`**: send the move_base action goal.
   3. On failure, **clear_costmaps + retry once 1 m back along the heading**.
   4. After arrival, **`_scan_for_boxes`**: spin in place at 0.7 rad/s for
      10 s (about one full revolution) so the camera covers every direction
      and the detector accumulates votes.
3. After all waypoints are visited, log `detected_counts`. The detector
   simultaneously latches the "digit with minimum count" to
   `/mission_planner/min_box_id`, which is the target for the upper phase.

---

## 3. Stairway Phase: Ramp Navigation

### 3.1 Pain Point

When the robot pitches up while climbing the ramp, the 2D laser sweeps
across the ramp surface and edges, which the ObstacleLayer marks as lethal.
After inflation the entire ramp becomes unreachable and the robot cannot
get up.

### 3.2 Solution

`run_stairway` does three things before executing
`waypoints_stairway.yaml`:

1. **Switch TEB to an "aggressive" parameter set** (via dynamic_reconfigure
   at runtime):
   * `min_obstacle_dist`: 0.45 -> 0.25
   * `inflation_dist`: 1.0 -> 0.6
   * `weight_obstacle`: 100 -> 50
   * `include_costmap_obstacles: false`, `include_dynamic_obstacles: false`
     -> TEB ignores costmap obstacles entirely and only follows the global
     plan.
2. **Disable obstacles_layer on both costmaps**:
   * `/move_base/global_costmap/obstacles_layer/enabled = false`
   * `/move_base/local_costmap/obstacles_layer/enabled = false`
   * Call `clear_costmaps` to wipe pre-switch dirty data.
3. **10 s auto-restore**: spawn a daemon thread that sleeps 10 s and then
   restores everything (re-enables obstacles_layer, re-enables TEB obstacle
   reading, calls `clear_costmaps` again). The ramp traversal only needs
   about 10 s; once the robot is on the upper level we want full obstacle
   avoidance back so any real box / barrel is avoided again. The `finally`
   block also restores once more, so it is safe even if the phase exits
   early due to an exception.

### 3.3 Barrier Handling

* Initial step: navigate back to the last lower-phase waypoint (near the
  barrier).
* Inside the stairway waypoint loop, at 10 Hz, check the distance from the
  robot to the `Construction Barrel` (read from `/gazebo/model_states` and
  transformed into the map frame). When distance < 3 m, publish
  `/cmd_unblock` (Bool=True) five times to make the simulation remove the
  barrier, then `clear_costmaps`.
* If move_base fails because the barrier is still present, an unblock is
  re-issued and the same goal is resent.
* After all stairway waypoints are visited, the robot stops at the last
  point.

---

## 4. Upper Phase: Room Identification and Entry

### 4.1 Waypoint Convention

`config/waypoints_upper.yaml` contains 8 points:

| Index | Role | Pairing |
|---|---|---|
| wp_0..wp_3 | 4 room-entrance scan points | wp_0<->wp_5, wp_1<->wp_6, wp_2<->wp_7, wp_3<->wp_4 |
| wp_4..wp_7 | 4 in-room target points | (paired by y-proximity) |

Pairing rule: `interior = entrances[(i + 1) % 4]`.

### 4.2 Flow (`run_upper_level`)

1. Load all 8 waypoints, split into entrances + interiors, pair them per
   the rule above.
2. **Reorder**: move the last pair (wp_3, wp_4) to the front so the robot
   first visits the room closest to the stairway exit. Visit order becomes
   `wp_3 -> wp_0 -> wp_1 -> wp_2`.
3. **Phase 1**: send_goal to the last stairway waypoint as the upper-phase
   start point.
4. Read `self.target_digits` (the list of target digits the lower-phase
   detector pushed; may contain more than one).
5. **Iterate entrances**:
   1. send_goal to the entrance (using the YAML-saved yaw, which faces into
      the room).
   2. Publish `/box_detector/reset` -> detector bumps scan_id (does not
      clear candidates).
   3. **`rotate_and_scan`**:
      * angular velocity **0.25 rad/s** (slow, to avoid missed detections
        when partially occluded)
      * total duration **30 s** (about 1.2 revolutions)
      * **micro-pause: 1 s of stillness every 4 s of spinning**, so the
        detector can accumulate several frames at the same view
      * any digit in `target_digits` showing up in `self.detected_counts`
        causes immediate return as `(True, matched_digit)`
   4. **Match**: send_goal to the paired interior point, mission complete
      on arrival.
   5. **No match**: continue to the next entrance.
6. **Multiple tied minimums**: if lower computed `target_digits=[2, 8]`,
   the loop scans rooms in patrol order `wp_3 -> wp_0 -> wp_1 -> wp_2` and
   **enters the first room whose visible digit matches either 2 or 8** —
   no hesitation, no full sweep.

### 4.3 Mission Complete

After a match, `run_upper_level` sends a goal to the interior point and
returns. `box_detector_3d` is still running:
* `min_box_locked=True`, no new min_box is published;
* every lower + upper candidate in `self.candidates` is **preserved**;
* markers use `lifetime=0`, so RViz keeps them on screen forever for manual
  inspection.

Pressing the "Stop" button cancels the move_base goal and disables
detection, but the markers remain frozen at the last frame for review.

---

## 5. Key Files

| File | Purpose |
|---|---|
| `scripts/box_detector_3d.py` | YOLO + depth + 3D dedup + vote counting + min_box publishing |
| `scripts/nav_test.py` | ZoneNavigator, three-phase mission orchestration |
| `scripts/mission_planner.py` | Backup state-machine implementation (main path uses zone_callback) |
| `launch/move_base.launch` | move_base node + Dijkstra/TEB configuration |
| `launch/navigation.launch` | Top entry point (map_server + AMCL + move_base + detector + zone_navigator) |
| `params/teb_local_planner_params.yaml` | TEB parameters (lower-phase defaults) |
| `params/costmap_common_params.yaml` | obstacle layer / inflation layer / footprint |
| `params/global_costmap_params.yaml` | global costmap dimensions |
| `params/local_costmap_params.yaml` | local costmap dimensions |
| `config/waypoints_lower.yaml` | lower-phase patrol waypoints |
| `config/waypoints_stairway.yaml` | stairway ramp waypoints |
| `config/waypoints_upper.yaml` | upper-phase entrances + interior targets (8 points) |
| `maps/final/2Dmap/final_map_v2.{pgm,yaml}` | static map |
| `models/Final.pt` | YOLO v8 model weights |

---

## 6. Sample Log

```
[ZoneNav] LOWER LEVEL EXPLORATION
[ZoneNav] Loaded 23 waypoints from file
[ZoneNav] Lower 1/23 [wp_0] -> (4.10, -0.58)
...
[ZoneNav] Box counts (camera 3D detection): {1: 4, 2: 1, 3: 3, 4: 2, 5: 4, 6: 3, 7: 4, 8: 1, 9: 3}
[ZoneNav] ANSWER: box #2 (count=1)
[ZoneNav] Target digits set to [2, 8]

[ZoneNav] STAIRWAY NAVIGATION
[ZoneNav] Stairway: switched TEB to modify-branch params
[ZoneNav] Stairway: disabled global obstacles_layer
[ZoneNav] Stairway: disabled local obstacles_layer
[ZoneNav] Robot 2.7m from barrier - UNBLOCKING!
[ZoneNav] Auto-restored global obstacles_layer (10s)
[ZoneNav] Auto-restored local obstacles_layer (10s)
[ZoneNav] Auto-restored TEB obstacle inclusion (10s)

[ZoneNav] UPPER LEVEL SEARCH
[ZoneNav] Looking for room containing any of [2, 8] (first match wins by patrol order)
[ZoneNav] Upper visit order: ['wp_3', 'wp_0', 'wp_1', 'wp_2']
[ZoneNav] Rotating to scan room at [wp_3]...
[ZoneNav] [wp_3] is not a target room (visible: {5: 1}), moving to next
[ZoneNav] Rotating to scan room at [wp_0]...
[ZoneNav] Target #2 found during scan!
[ZoneNav] Matched #2 at [wp_0] -> entering paired interior [wp_5] (25.89, 4.58)
[ZoneNav] Arrived at target box #2. Mission complete.
```
