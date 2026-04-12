#!/usr/bin/env python3
"""
Zone-based Navigation with Coverage Path Planning

Controlled by RViz panel buttons:
  - "Lower Level": coverage patrol of lower level, stop at barrier
  - "Stairway": unblock barrier, navigate ramp to upper level
  - "Upper Level": coverage patrol of upper level
  - "Stop": cancel current navigation

Subscribes: /nav_zone/command (String)
Box detection runs via separate box_detector.py node (OpenCV).
"""

import rospy
import actionlib
import math
import yaml
import rospkg
import threading
import tf2_ros
import tf2_geometry_msgs
import dynamic_reconfigure.client

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool, Empty
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty as EmptySrv


class ZoneNavigator:
    def __init__(self):
        rospy.init_node('zone_navigator', anonymous=False)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Move base
        rospy.loginfo("[ZoneNav] Waiting for move_base...")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server(rospy.Duration(30.0))
        rospy.loginfo("[ZoneNav] move_base connected!")

        # Publishers
        self.pub_unblock = rospy.Publisher('/cmd_unblock', Bool, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_det_reset = rospy.Publisher('/box_detector/reset', Empty, queue_size=1)
        self.pub_initialpose = rospy.Publisher(
            '/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)

        # Localization calibration runs once before the first zone command
        self.calibrated = False

        # Waypoint file paths (one per zone)
        rospack = rospkg.RosPack()
        config_dir = rospack.get_path('me5413_team_solution') + '/config'
        self.waypoint_files = {
            'lower':    config_dir + '/waypoints_lower.yaml',
            'stairway': config_dir + '/waypoints_stairway.yaml',
            'upper':    config_dir + '/waypoints_upper.yaml',
        }
        rospy.loginfo("[ZoneNav] Waypoint files: %s", self.waypoint_files)

        # State
        self.running = False
        self.stop_requested = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.cone_map_pos = None

        # Global costmap (for goal-adjustment when a box sits on a waypoint)
        self.global_costmap = None
        self.sub_costmap = rospy.Subscriber(
            '/move_base/global_costmap/costmap', OccupancyGrid,
            self.costmap_callback, queue_size=1)

        # Camera detection results (from box_detector_3d)
        self.detected_counts = {}
        self.sub_box_counts = rospy.Subscriber(
            '/box_detector/counts', String, self.box_counts_callback)

        # Target digits from lower-phase answer (latched topic). May contain
        # more than one digit when several digits tie for the minimum count.
        self.target_digits = []
        self.sub_min_box = rospy.Subscriber(
            '/mission_planner/min_box_id', String, self.min_box_callback)

        # Subscribe FIRST so we never miss commands
        self.sub_zone = rospy.Subscriber(
            '/nav_zone/command', String, self.zone_callback)
        rospy.loginfo("[ZoneNav] Subscribed to /nav_zone/command")

        # Load ground truth in background (non-blocking)
        t = threading.Thread(target=self._init_ground_truth, daemon=True)
        t.start()

        rospy.loginfo("[ZoneNav] Ready. Use RViz panel buttons to start.")

    def costmap_callback(self, msg):
        self.global_costmap = msg

    def _costmap_cost(self, x, y):
        """Return the global-costmap cell cost at (x, y) in map frame, or
        None if the point is outside the costmap or the costmap is not yet
        available. 0 = free, 100 = lethal, -1 = unknown."""
        cm = self.global_costmap
        if cm is None:
            return None
        res = cm.info.resolution
        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if mx < 0 or my < 0 or mx >= cm.info.width or my >= cm.info.height:
            return None
        return cm.data[my * cm.info.width + mx]

    def _adjust_goal(self, x, y, yaw):
        """If the waypoint is occupied by a dynamically-spawned box, shift it
        to a nearby free cell. Scans 8 directions at 3 increasing radii.
        Returns (x, y) - unchanged if the original is free or no alternative
        was found."""
        cost = self._costmap_cost(x, y)
        if cost is None or cost < 80:
            return x, y  # free (or costmap not ready)
        rospy.loginfo("[ZoneNav] Waypoint (%.2f,%.2f) is blocked (cost=%s), searching nearby",
                      x, y, str(cost))
        for r in (0.7, 1.2, 1.8):
            for k in range(8):
                ang = k * math.pi / 4.0
                nx = x + r * math.cos(ang)
                ny = y + r * math.sin(ang)
                c = self._costmap_cost(nx, ny)
                if c is not None and c < 50:
                    rospy.loginfo("[ZoneNav]   -> adjusted to (%.2f,%.2f) r=%.1fm cost=%d",
                                  nx, ny, r, c)
                    return nx, ny
        rospy.logwarn("[ZoneNav]   -> no free cell found, using original")
        return x, y

    def _scan_for_boxes(self, duration=10.0, angular_vel=0.7):
        """Spin 360 degrees in place so the camera covers all directions,
        letting box_detector_3d accumulate votes. No early-exit -- the point
        is coverage, not match. Returns when duration expires or stop
        requested."""
        twist = Twist()
        twist.angular.z = angular_vel
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        while not rospy.is_shutdown() and not self.stop_requested:
            if (rospy.Time.now() - start).to_sec() > duration:
                break
            self.pub_cmd_vel.publish(twist)
            rate.sleep()
        self.pub_cmd_vel.publish(Twist())

    def _clear_costmaps(self):
        try:
            svc = rospy.ServiceProxy('/move_base/clear_costmaps', EmptySrv)
            svc()
            rospy.loginfo("[ZoneNav] Costmaps cleared")
        except Exception as e:
            rospy.logwarn("[ZoneNav] clear_costmaps failed: %s", str(e))

    def _get_jackal_gazebo_pose(self):
        """Pull the Jackal ground-truth pose from /gazebo/model_states and
        convert it into the map frame (via the world->map TF that the world
        plugin publishes). Returns (x, y, yaw) in map frame or None."""
        try:
            msg = rospy.wait_for_message(
                '/gazebo/model_states', ModelStates, timeout=3.0)
        except Exception:
            return None
        for i, name in enumerate(msg.name):
            if name in ('jackal', '/', 'jackal::base_link'):
                p = msg.pose[i]
                wx, wy = p.position.x, p.position.y
                mx, my = self.world_to_map(wx, wy)
                if mx is None:
                    return None
                # yaw from quaternion (z,w only -- pitch/roll are 0 on flat floor)
                qz = p.orientation.z
                qw = p.orientation.w
                yaw = 2.0 * math.atan2(qz, qw)
                return mx, my, yaw
        return None

    def calibrate_localization(self):
        """Fix the AMCL initial pose before navigation starts. RViz sometimes
        shows a small offset between the laser scan and the static map because
        AMCL hasn't converged. We:
          1. Inject the ground-truth pose from Gazebo into /initialpose so
             AMCL particles collapse onto the correct location.
          2. Spin slowly in place so the laser sweeps 360 degrees of the
             environment, giving AMCL strong observations to refine on.
          3. Clear costmaps so the old (offset) laser hits are forgotten."""
        if self.calibrated:
            return
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] LOCALIZATION CALIBRATION")
        rospy.loginfo("=" * 50)

        # --- Step 1: publish ground truth to /initialpose ---
        pose = self._get_jackal_gazebo_pose()
        if pose is not None:
            mx, my, yaw = pose
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = rospy.Time.now()
            msg.pose.pose.position.x = mx
            msg.pose.pose.position.y = my
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            # Small covariance (x=y=0.25m, yaw=~0.07rad) so AMCL tightens
            # particles around this estimate rather than scattering them.
            cov = [0.0] * 36
            cov[0] = 0.05      # x-x
            cov[7] = 0.05      # y-y
            cov[35] = 0.0685   # yaw-yaw
            msg.pose.covariance = cov
            # Publish a few times so AMCL definitely sees it
            for _ in range(5):
                self.pub_initialpose.publish(msg)
                rospy.sleep(0.1)
            rospy.loginfo("[ZoneNav] Initial pose set: (%.2f, %.2f, yaw=%.2f)",
                          mx, my, yaw)
        else:
            rospy.logwarn("[ZoneNav] Could not get Jackal pose from Gazebo, "
                          "skipping initial pose injection")

        # --- Step 2: spin slowly so AMCL gets 360deg laser observations ---
        rospy.loginfo("[ZoneNav] Spinning to refine AMCL particles...")
        twist = Twist()
        twist.angular.z = 0.5
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        while not rospy.is_shutdown() and not self.stop_requested:
            if (rospy.Time.now() - start).to_sec() > 13.0:  # ~one full turn at 0.5 rad/s
                break
            self.pub_cmd_vel.publish(twist)
            rate.sleep()
        self.pub_cmd_vel.publish(Twist())

        # --- Step 3: clear costmaps so stale (pre-calibration) hits vanish ---
        rospy.sleep(0.3)
        self._clear_costmaps()

        self.calibrated = True
        rospy.loginfo("[ZoneNav] Calibration complete.")

    def box_counts_callback(self, msg):
        """Receive detected box counts from box_detector_3d."""
        try:
            self.detected_counts = eval(msg.data) if msg.data else {}
        except Exception:
            pass

    def min_box_callback(self, msg):
        """Lower-phase answer: digits tied for minimum count, comma-separated.
        e.g. '8' (single) or '2,8' (two-way tie)."""
        try:
            parts = [p.strip() for p in msg.data.strip().split(',') if p.strip()]
            self.target_digits = [int(p) for p in parts]
            rospy.loginfo("[ZoneNav] Target digits set to %s", self.target_digits)
        except ValueError:
            pass

    def rotate_and_scan(self, target_digits, duration=30.0, angular_vel=0.25,
                        pause_every=4.0, pause_duration=1.0):
        """Spin slowly in place, pausing periodically so the YOLO 3D detector
        has time to confirm a candidate even when the box is partially
        occluded. Returns (found, matched_digit). matched_digit is the first
        digit in target_digits that showed up in detector's current counts;
        None if no match.

        Slower angular_vel + periodic pauses give the detector several frames
        on the same view, fixing the issue where a fast spin (0.4 rad/s) was
        skipping past partially-blocked boxes."""
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        last_pause = rospy.Time.now()
        targets = list(target_digits) if target_digits else []
        spinning = True
        while not rospy.is_shutdown() and not self.stop_requested:
            if (rospy.Time.now() - start).to_sec() > duration:
                break

            now = rospy.Time.now()
            elapsed_since_pause = (now - last_pause).to_sec()
            if spinning and elapsed_since_pause >= pause_every:
                # micro-pause: stop spinning and stare for pause_duration
                self.pub_cmd_vel.publish(Twist())
                spinning = False
                pause_start = now
            elif not spinning and (now - pause_start).to_sec() >= pause_duration:
                spinning = True
                last_pause = now

            if spinning:
                twist = Twist()
                twist.angular.z = angular_vel
                self.pub_cmd_vel.publish(twist)

            for d in targets:
                if d in self.detected_counts:
                    rospy.loginfo("[ZoneNav] Target #%d found during scan!", d)
                    self.pub_cmd_vel.publish(Twist())
                    return True, d
            rate.sleep()
        self.pub_cmd_vel.publish(Twist())
        return False, None

    def _init_ground_truth(self):
        """Load barrier position from Gazebo in background, doesn't block startup."""
        rospy.sleep(3.0)
        self.robot_x, self.robot_y = self._get_robot_pose()
        rospy.loginfo("[ZoneNav] Robot at map (%.2f, %.2f)",
                      self.robot_x, self.robot_y)
        try:
            model_msg = rospy.wait_for_message(
                '/gazebo/model_states', ModelStates, timeout=10.0)
            self._parse_barrier(model_msg)
        except Exception:
            rospy.logwarn("[ZoneNav] No model states, skipping barrier detection")

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            rospy.logwarn("[ZoneNav] Cannot get robot pose (map frame not ready)")
            return 0.0, 0.0

    def _parse_barrier(self, msg):
        """Extract barrier (Construction Barrel) position from Gazebo models."""
        self.cone_world_pos = None
        for i, name in enumerate(msg.name):
            if name == 'Construction Barrel':
                pose = msg.pose[i]
                self.cone_world_pos = (pose.position.x, pose.position.y)
                break

        # Convert cone to map frame
        self.cone_map_pos = None
        if self.cone_world_pos:
            cx, cy = self.world_to_map(*self.cone_world_pos)
            if cx is not None:
                self.cone_map_pos = (cx, cy)
                rospy.loginfo("[ZoneNav] Barrier at map (%.1f, %.1f)", cx, cy)

    def world_to_map(self, wx, wy):
        p = PoseStamped()
        p.header.frame_id = "world"
        p.header.stamp = rospy.Time(0)
        p.pose.position.x = wx
        p.pose.position.y = wy
        p.pose.orientation.w = 1.0
        try:
            pm = self.tf_buffer.transform(p, "map", rospy.Duration(2.0))
            return pm.pose.position.x, pm.pose.position.y
        except Exception:
            rospy.logwarn("[ZoneNav] world->map TF not available")
            return None, None

    def load_waypoints(self, zone='lower', use_saved_yaw=False):
        """Load waypoints from zone-specific YAML file.
        If use_saved_yaw is True, the yaw field from YAML is used as-is
        (needed for upper-level room entrances that face into the rooms).
        Otherwise yaw is computed as the direction of travel."""
        try:
            wp_file = self.waypoint_files.get(zone, self.waypoint_files['lower'])
            rospy.loginfo("[ZoneNav] Loading waypoints from: %s", wp_file)
            with open(wp_file, 'r') as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, list):
                rospy.logwarn("[ZoneNav] No waypoints in file")
                return []
            waypoints = []
            for wp in data:
                x = float(wp['x'])
                y = float(wp['y'])
                name = wp.get('name', 'wp')
                yaw = float(wp.get('yaw', 0.0))
                waypoints.append((x, y, yaw, name))

            if not use_saved_yaw:
                # Compute yaw: face from previous point to current point
                for i in range(1, len(waypoints)):
                    x_prev, y_prev, _, _ = waypoints[i - 1]
                    x_cur, y_cur, _, name_cur = waypoints[i]
                    yaw = math.atan2(y_cur - y_prev, x_cur - x_prev)
                    waypoints[i] = (x_cur, y_cur, yaw, name_cur)
                # First point: use yaw toward second point
                if len(waypoints) >= 2:
                    x1, y1, _, name1 = waypoints[0]
                    x2, y2, _, _ = waypoints[1]
                    yaw = math.atan2(y2 - y1, x2 - x1)
                    waypoints[0] = (x1, y1, yaw, name1)

            rospy.loginfo("[ZoneNav] Loaded %d waypoints from file", len(waypoints))
            return waypoints
        except Exception as e:
            rospy.logerr("[ZoneNav] Failed to load waypoints: %s", str(e))
            return []

    def send_goal(self, x, y, yaw=0.0, timeout=35.0):
        if self.stop_requested:
            return False
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)

        rospy.loginfo("[ZoneNav] Goal: map (%.2f, %.2f)", x, y)
        self.client.send_goal(goal)

        # Poll for result, checking stop flag
        rate = rospy.Rate(5)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.stop_requested:
                self.client.cancel_goal()
                rospy.loginfo("[ZoneNav] Goal cancelled (stop)")
                return False
            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("[ZoneNav] Reached!")
                return True
            if state in [GoalStatus.ABORTED, GoalStatus.REJECTED,
                         GoalStatus.PREEMPTED, GoalStatus.LOST]:
                rospy.logwarn("[ZoneNav] Failed (state=%d)", state)
                return False
            if (rospy.Time.now() - start).to_sec() > timeout:
                self.client.cancel_goal()
                rospy.logwarn("[ZoneNav] Timeout")
                return False
            rate.sleep()
        return False

    def navigate_waypoints(self, waypoints, label=""):
        reached = 0
        for i, (x, y, yaw) in enumerate(waypoints):
            if rospy.is_shutdown() or self.stop_requested:
                break
            rospy.loginfo("[ZoneNav] %s %d/%d", label, i+1, len(waypoints))
            if self.send_goal(x, y, yaw):
                reached += 1
            rospy.sleep(0.3)
        rospy.loginfo("[ZoneNav] %s: %d/%d reached", label, reached, len(waypoints))
        return reached

    # =========================================================
    # ZONE HANDLERS
    # =========================================================

    def run_lower_level(self):
        """Navigate lower level using saved waypoints in order."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] LOWER LEVEL EXPLORATION")
        rospy.loginfo("=" * 50)

        # Load waypoints from lower-level file
        waypoints = self.load_waypoints('lower')
        if not waypoints:
            rospy.logerr("[ZoneNav] No lower waypoints! Save some with the panel first.")
            return

        # Navigate each waypoint in saved order (yaw already computed in load_waypoints)
        reached = 0
        for i, (x, y, yaw, name) in enumerate(waypoints):
            if rospy.is_shutdown() or self.stop_requested:
                break
            # If a dynamically-spawned box is sitting on this waypoint, offset it.
            gx, gy = self._adjust_goal(x, y, yaw)
            rospy.loginfo("[ZoneNav] Lower %d/%d [%s] -> (%.2f, %.2f)",
                          i + 1, len(waypoints), name, gx, gy)
            ok = self.send_goal(gx, gy, yaw)
            if not ok:
                # Goal failed: clear costmaps and retry 1m back along approach direction
                rospy.logwarn("[ZoneNav] Goal failed, clearing costmaps and retrying")
                self._clear_costmaps()
                rospy.sleep(0.5)
                retry_x = gx - 1.0 * math.cos(yaw)
                retry_y = gy - 1.0 * math.sin(yaw)
                rospy.logwarn("[ZoneNav] Retrying 1m back -> (%.2f, %.2f)", retry_x, retry_y)
                ok = self.send_goal(retry_x, retry_y, yaw, timeout=20.0)
            if ok:
                reached += 1
                # Rotate in place so the camera catches boxes in every direction
                rospy.loginfo("[ZoneNav] Scanning 360 at [%s]", name)
                self._scan_for_boxes(duration=10.0, angular_vel=0.7)

        rospy.loginfo("[ZoneNav] Lower level: %d/%d waypoints reached", reached, len(waypoints))

        # Print camera detection results (3D deduplicated)
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] Box counts (camera 3D detection): %s", self.detected_counts)
        if self.detected_counts:
            min_box = min(self.detected_counts, key=self.detected_counts.get)
            rospy.loginfo("[ZoneNav] ANSWER: box #%d (count=%d)",
                          min_box, self.detected_counts[min_box])
        else:
            rospy.logwarn("[ZoneNav] No boxes detected by camera yet.")
        rospy.loginfo("=" * 50)

    def run_stairway(self):
        """Go to barrier area, unblock when close, navigate stairway points, then auto up."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] STAIRWAY NAVIGATION")
        rospy.loginfo("=" * 50)

        # Load stairway waypoints from dedicated file
        stairway_waypoints = self.load_waypoints('stairway')
        if not stairway_waypoints:
            rospy.logerr("[ZoneNav] No stairway waypoints! Save some with the panel first.")
            return

        # Load lower waypoints to get last point (near barrier)
        lower_waypoints = self.load_waypoints('lower')

        # === Phase 1: Go to last lower level waypoint (near barrier) ===
        if lower_waypoints:
            last_lower = lower_waypoints[-1]
            x, y, yaw, name = last_lower
            rospy.loginfo("[ZoneNav] Going to last lower waypoint [%s] -> (%.2f, %.2f)", name, x, y)
            self.send_goal(x, y, yaw, timeout=60.0)

        if self.stop_requested:
            return

        # Get barrier position
        barrier_x = self.cone_map_pos[0] if self.cone_map_pos else 7.7
        barrier_y = self.cone_map_pos[1] if self.cone_map_pos else -1.9
        unblock_distance = 3.0
        barrier_removed = False

        # === Phase 2: Switch TEB to modify-branch params for stairway ===
        teb_client = None
        local_teb_params = {
            'min_obstacle_dist': 0.45,
            'inflation_dist': 1.0,
            'dynamic_obstacle_inflation_dist': 1.0,
            'costmap_converter_rate': 10,
            'weight_obstacle': 100.0,
            'weight_inflation': 1.0,
            'weight_dynamic_obstacle': 50.0,
            'weight_dynamic_obstacle_inflation': 1.0,
            'include_costmap_obstacles': True,
            'include_dynamic_obstacles': True,
        }
        modify_teb_params = {
            'min_obstacle_dist': 0.25,
            'inflation_dist': 0.6,
            'dynamic_obstacle_inflation_dist': 0.6,
            'costmap_converter_rate': 5,
            'weight_obstacle': 50.0,
            'weight_inflation': 0.2,
            'weight_dynamic_obstacle': 10.0,
            'weight_dynamic_obstacle_inflation': 0.2,
            # Static-map-only mode: TEB ignores live obstacles entirely.
            # Ramp surface returns and any dynamic blockers won't be optimized against.
            'include_costmap_obstacles': False,
            'include_dynamic_obstacles': False,
        }
        try:
            teb_client = dynamic_reconfigure.client.Client(
                '/move_base/TebLocalPlannerROS', timeout=5)
            applied = teb_client.update_configuration(modify_teb_params)
            rospy.loginfo("[ZoneNav] Stairway: switched TEB to modify-branch params")
            for k in modify_teb_params:
                rospy.loginfo("[ZoneNav]   %s: %s -> %s (actual=%s)",
                              k, local_teb_params[k], modify_teb_params[k],
                              applied.get(k, '?'))
        except Exception as e:
            rospy.logerr("[ZoneNav] TEB reconfigure failed: %s", e)
            teb_client = None

        # Disable the obstacle layer on both costmaps so the planner only
        # sees the static map. The 2D laser otherwise paints the ramp surface
        # as lethal when the robot pitches up.
        global_obs_client = None
        local_obs_client = None
        try:
            global_obs_client = dynamic_reconfigure.client.Client(
                '/move_base/global_costmap/obstacles_layer', timeout=5)
            global_obs_client.update_configuration({'enabled': False})
            rospy.loginfo("[ZoneNav] Stairway: disabled global obstacles_layer")
        except Exception as e:
            rospy.logwarn("[ZoneNav] global obstacles_layer reconfigure failed: %s", e)
            global_obs_client = None
        try:
            local_obs_client = dynamic_reconfigure.client.Client(
                '/move_base/local_costmap/obstacles_layer', timeout=5)
            local_obs_client.update_configuration({'enabled': False})
            rospy.loginfo("[ZoneNav] Stairway: disabled local obstacles_layer")
        except Exception as e:
            rospy.logwarn("[ZoneNav] local obstacles_layer reconfigure failed: %s", e)
            local_obs_client = None
        # Wipe whatever was already marked before we switched modes.
        self._clear_costmaps()

        # Auto-restore obstacle avoidance after 10s. The ramp traversal only
        # needs ~10s of "static-map-only" navigation; once the robot is on
        # the upper level we want full obstacle avoidance back so any real
        # box / barrel is avoided again.
        restore_state = {'done': False}

        def _auto_restore_obstacles():
            rospy.sleep(10.0)
            if restore_state['done']:
                return
            restore_state['done'] = True
            if global_obs_client is not None:
                try:
                    global_obs_client.update_configuration({'enabled': True})
                    rospy.loginfo("[ZoneNav] Auto-restored global obstacles_layer (10s)")
                except Exception:
                    pass
            if local_obs_client is not None:
                try:
                    local_obs_client.update_configuration({'enabled': True})
                    rospy.loginfo("[ZoneNav] Auto-restored local obstacles_layer (10s)")
                except Exception:
                    pass
            if teb_client is not None:
                try:
                    teb_client.update_configuration({
                        'include_costmap_obstacles': True,
                        'include_dynamic_obstacles': True,
                    })
                    rospy.loginfo("[ZoneNav] Auto-restored TEB obstacle inclusion (10s)")
                except Exception:
                    pass
            self._clear_costmaps()

        threading.Thread(target=_auto_restore_obstacles, daemon=True).start()

        try:
            # === Phase 3: Navigate stairway waypoints, unblock when near barrier ===
            for i, (sx, sy, syaw, sname) in enumerate(stairway_waypoints):
                if rospy.is_shutdown() or self.stop_requested:
                    break

                rospy.loginfo("[ZoneNav] Stairway %d/%d [%s] -> (%.2f, %.2f)",
                              i + 1, len(stairway_waypoints), sname, sx, sy)

                # Send goal
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = sx
                goal.target_pose.pose.position.y = sy
                goal.target_pose.pose.orientation.z = math.sin(syaw / 2.0)
                goal.target_pose.pose.orientation.w = math.cos(syaw / 2.0)
                self.client.send_goal(goal)

                # Poll for result, check barrier distance
                rate = rospy.Rate(10)
                start = rospy.Time.now()
                timeout = 300.0
                while not rospy.is_shutdown():
                    if self.stop_requested:
                        self.client.cancel_goal()
                        break

                    # Check if we should unblock barrier
                    if not barrier_removed:
                        rx, ry = self._get_robot_pose()
                        dist = math.sqrt((rx - barrier_x)**2 + (ry - barrier_y)**2)
                        if dist < unblock_distance:
                            rospy.loginfo("[ZoneNav] Robot %.1fm from barrier - UNBLOCKING!", dist)
                            msg = Bool()
                            msg.data = True
                            for _ in range(5):
                                self.pub_unblock.publish(msg)
                                rospy.sleep(0.05)
                            barrier_removed = True
                            rospy.sleep(0.3)
                            try:
                                clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', EmptySrv)
                                clear_costmaps()
                                rospy.loginfo("[ZoneNav] Costmaps cleared!")
                            except Exception:
                                pass

                    state = self.client.get_state()
                    if state == GoalStatus.SUCCEEDED:
                        rospy.loginfo("[ZoneNav] Reached [%s]!", sname)
                        break
                    if state in [GoalStatus.ABORTED, GoalStatus.REJECTED,
                                 GoalStatus.PREEMPTED, GoalStatus.LOST]:
                        if not barrier_removed:
                            rospy.logwarn("[ZoneNav] Blocked by barrier, unblocking and retrying...")
                            msg = Bool()
                            msg.data = True
                            for _ in range(5):
                                self.pub_unblock.publish(msg)
                                rospy.sleep(0.05)
                            barrier_removed = True
                            rospy.sleep(0.5)
                            try:
                                clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', EmptySrv)
                                clear_costmaps()
                            except Exception:
                                pass
                            self.client.send_goal(goal)
                        else:
                            rospy.logwarn("[ZoneNav] Failed [%s] (state=%d), continuing...", sname, state)
                            break
                    if (rospy.Time.now() - start).to_sec() > timeout:
                        self.client.cancel_goal()
                        rospy.logwarn("[ZoneNav] Timeout [%s]", sname)
                        break
                    rate.sleep()
        finally:
            # Mark restore as done so the 10s auto-restore thread (if still
            # sleeping) becomes a no-op when it wakes up.
            restore_state['done'] = True
            if teb_client is not None:
                try:
                    teb_client.update_configuration(local_teb_params)
                    rospy.loginfo("[ZoneNav] TEB restored to local params")
                except Exception:
                    pass
            if global_obs_client is not None:
                try:
                    global_obs_client.update_configuration({'enabled': True})
                    rospy.loginfo("[ZoneNav] Re-enabled global obstacles_layer")
                except Exception:
                    pass
            if local_obs_client is not None:
                try:
                    local_obs_client.update_configuration({'enabled': True})
                    rospy.loginfo("[ZoneNav] Re-enabled local obstacles_layer")
                except Exception:
                    pass
            self._clear_costmaps()

        rospy.loginfo("[ZoneNav] Stairway finished, stopped at last waypoint.")

    def run_upper_level(self):
        """Visit each room-entrance waypoint, spin to scan, and enter the
        room whose visible box digit matches the lower-phase answer."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] UPPER LEVEL SEARCH")
        rospy.loginfo("=" * 50)

        all_waypoints = self.load_waypoints('upper', use_saved_yaw=True)
        if not all_waypoints:
            rospy.logwarn("[ZoneNav] No upper waypoints! Save some with the panel first.")
            return

        # Layout convention in waypoints_upper.yaml:
        #   wp_0..wp_3 -> room entrance scan points (4 rooms)
        #   wp_4..wp_7 -> matching in-room target points
        # Pairing by y-proximity (interior shifts one position forward):
        #   wp_0 <-> wp_5, wp_1 <-> wp_6, wp_2 <-> wp_7, wp_3 <-> wp_4
        # i.e. interiors[(i + 1) % 4]
        if len(all_waypoints) < 4:
            rospy.logwarn("[ZoneNav] Need at least 4 upper waypoints, got %d",
                          len(all_waypoints))
            return
        entrances = all_waypoints[:4]
        interiors = all_waypoints[4:8] if len(all_waypoints) >= 8 else []

        # Pair them up so reordering keeps the (entrance, interior) link intact.
        pairs = []
        for i, ent in enumerate(entrances):
            if len(interiors) == 4:
                interior = interiors[(i + 1) % 4]
            else:
                interior = None
            pairs.append((ent, interior))

        # Reorder: visit the last entrance first (closest to stairway exit),
        # then the rest in original order. Interior partners ride along.
        if len(pairs) >= 2:
            pairs = [pairs[-1]] + pairs[:-1]
        rospy.loginfo("[ZoneNav] Upper visit order: %s",
                      [p[0][3] for p in pairs])

        # === Phase 1: Go to last stairway waypoint as the upper-level start ===
        stairway_waypoints = self.load_waypoints('stairway')
        if stairway_waypoints:
            sx, sy, syaw, sname = stairway_waypoints[-1]
            rospy.loginfo("[ZoneNav] Phase 1: going to last stairway waypoint [%s] -> (%.2f, %.2f)",
                          sname, sx, sy)
            self.send_goal(sx, sy, syaw, timeout=120.0)

        if self.stop_requested:
            return

        targets = list(self.target_digits)
        if not targets:
            rospy.logwarn("[ZoneNav] No target digits from lower phase yet, "
                          "will still scan each room but can't match.")
        else:
            rospy.loginfo("[ZoneNav] Looking for room containing any of %s "
                          "(first match wins by patrol order)", targets)

        enter_forward_dist = 3.0  # fallback only, when no interior point is paired

        for i, (entrance, interior) in enumerate(pairs):
            if rospy.is_shutdown() or self.stop_requested:
                break

            x, y, yaw, name = entrance
            rospy.loginfo("[ZoneNav] Upper %d/%d entrance [%s] -> (%.2f, %.2f)",
                          i + 1, len(pairs), name, x, y)

            if not self.send_goal(x, y, yaw, timeout=90.0):
                rospy.logwarn("[ZoneNav] Failed to reach [%s], skipping", name)
                continue

            if self.stop_requested:
                break

            # Bump the detector's scan window so this room's match is isolated
            # from previous rooms (candidates themselves are kept for marker
            # display — see box_detector_3d.reset_callback).
            self.pub_det_reset.publish(Empty())
            rospy.sleep(0.5)

            rospy.loginfo("[ZoneNav] Rotating to scan room at [%s]...", name)
            found, matched_digit = self.rotate_and_scan(
                targets, duration=30.0, angular_vel=0.25,
                pause_every=4.0, pause_duration=1.0)

            if found:
                if interior is not None:
                    ix, iy, iyaw, iname = interior
                    rospy.loginfo("[ZoneNav] Matched #%d at [%s] -> entering paired interior [%s] (%.2f, %.2f)",
                                  matched_digit, name, iname, ix, iy)
                    self.send_goal(ix, iy, iyaw, timeout=60.0)
                else:
                    enter_x = x + enter_forward_dist * math.cos(yaw)
                    enter_y = y + enter_forward_dist * math.sin(yaw)
                    rospy.logwarn("[ZoneNav] No paired interior for [%s], "
                                  "falling back to %.1fm forward step", name, enter_forward_dist)
                    self.send_goal(enter_x, enter_y, yaw, timeout=45.0)
                rospy.loginfo("[ZoneNav] Arrived at target box #%d. Mission complete.",
                              matched_digit)
                return
            else:
                rospy.loginfo("[ZoneNav] [%s] is not a target room (visible: %s), "
                              "moving to next", name, str(self.detected_counts))

        rospy.logwarn("[ZoneNav] Upper search finished without matching any of %s",
                      str(targets))

    # =========================================================

    def _run_zone(self, cmd):
        """Execute zone navigation in a separate thread."""
        try:
            # One-shot localization calibration before the very first task
            if not self.calibrated and cmd in ("lower", "stairway", "upper"):
                self.calibrate_localization()
                if self.stop_requested:
                    return
            if cmd == "lower":
                self.run_lower_level()
            elif cmd == "stairway":
                self.run_stairway()
            elif cmd == "upper":
                self.run_upper_level()
        except Exception as e:
            rospy.logerr("[ZoneNav] Error: %s", str(e))
        finally:
            self.running = False
            rospy.loginfo("[ZoneNav] Zone '%s' finished.", cmd)

    def zone_callback(self, msg):
        """Handle zone command from panel."""
        cmd = msg.data.strip().lower()
        rospy.loginfo("[ZoneNav] Received command: '%s'", cmd)

        if cmd == "stop":
            self.stop_requested = True
            self.client.cancel_all_goals()
            stop_msg = Twist()
            self.pub_cmd_vel.publish(stop_msg)
            rospy.loginfo("[ZoneNav] STOPPED")
            self.running = False
            return

        if self.running:
            # If currently running, stop first then start new zone
            rospy.loginfo("[ZoneNav] Stopping current task, starting '%s'", cmd)
            self.stop_requested = True
            self.client.cancel_all_goals()
            rospy.sleep(1.0)
            self.running = False

        self.stop_requested = False
        self.running = True

        # Run in a thread so callback returns immediately
        t = threading.Thread(target=self._run_zone, args=(cmd,))
        t.daemon = True
        t.start()


if __name__ == '__main__':
    try:
        node = ZoneNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
