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

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, Twist
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

        # Camera detection results (from box_detector_3d)
        self.detected_counts = {}
        self.sub_box_counts = rospy.Subscriber(
            '/box_detector/counts', String, self.box_counts_callback)

        # Target digit from lower-phase answer (latched topic)
        self.target_digit = None
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

    def box_counts_callback(self, msg):
        """Receive detected box counts from box_detector_3d."""
        try:
            self.detected_counts = eval(msg.data) if msg.data else {}
        except Exception:
            pass

    def min_box_callback(self, msg):
        """Lower-phase answer: digit that appeared least often."""
        try:
            self.target_digit = int(msg.data.strip())
            rospy.loginfo("[ZoneNav] Target digit set to #%d", self.target_digit)
        except ValueError:
            pass

    def rotate_and_scan(self, target_digit, duration=16.0, angular_vel=0.4):
        """Spin in place. Return True as soon as target_digit appears in
        the detector's confirmed counts; False after a full rotation."""
        twist = Twist()
        twist.angular.z = angular_vel
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        found = False
        while not rospy.is_shutdown() and not self.stop_requested:
            if (rospy.Time.now() - start).to_sec() > duration:
                break
            self.pub_cmd_vel.publish(twist)
            if target_digit is not None and target_digit in self.detected_counts:
                rospy.loginfo("[ZoneNav] Target #%d found during scan!", target_digit)
                found = True
                break
            rate.sleep()
        self.pub_cmd_vel.publish(Twist())
        return found

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
            rospy.loginfo("[ZoneNav] Lower %d/%d [%s] -> (%.2f, %.2f)",
                          i + 1, len(waypoints), name, x, y)
            if self.send_goal(x, y, yaw):
                reached += 1
            else:
                # Goal failed (possibly blocked by a box), try 1m back along approach direction
                retry_x = x - 1.0 * math.cos(yaw)
                retry_y = y - 1.0 * math.sin(yaw)
                rospy.logwarn("[ZoneNav] Retrying 1m back -> (%.2f, %.2f)", retry_x, retry_y)
                if self.send_goal(retry_x, retry_y, yaw, timeout=20.0):
                    reached += 1

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

        rospy.loginfo("[ZoneNav] Stairway finished, stopped at last waypoint.")

    def run_upper_level(self):
        """Visit each room-entrance waypoint, spin to scan, and enter the
        room whose visible box digit matches the lower-phase answer."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("[ZoneNav] UPPER LEVEL SEARCH")
        rospy.loginfo("=" * 50)

        waypoints = self.load_waypoints('upper', use_saved_yaw=True)
        if not waypoints:
            rospy.logwarn("[ZoneNav] No upper waypoints! Save some with the panel first.")
            return

        # Reorder: visit the most-recently-saved waypoint first (closest to
        # stairway exit), then the rest in original order.
        if len(waypoints) >= 2:
            waypoints = [waypoints[-1]] + waypoints[:-1]
            rospy.loginfo("[ZoneNav] Upper visit order: %s",
                          [w[3] for w in waypoints])

        # === Phase 1: Go to last stairway waypoint as the upper-level start ===
        stairway_waypoints = self.load_waypoints('stairway')
        if stairway_waypoints:
            sx, sy, syaw, sname = stairway_waypoints[-1]
            rospy.loginfo("[ZoneNav] Phase 1: going to last stairway waypoint [%s] -> (%.2f, %.2f)",
                          sname, sx, sy)
            self.send_goal(sx, sy, syaw, timeout=120.0)

        if self.stop_requested:
            return

        target = self.target_digit
        if target is None:
            rospy.logwarn("[ZoneNav] No target digit from lower phase yet, "
                          "will still scan each room but can't match.")
        else:
            rospy.loginfo("[ZoneNav] Looking for room containing box #%d", target)

        enter_forward_dist = 2.0  # how far into the room after confirming match

        for i, (x, y, yaw, name) in enumerate(waypoints):
            if rospy.is_shutdown() or self.stop_requested:
                break

            rospy.loginfo("[ZoneNav] Upper %d/%d [%s] -> (%.2f, %.2f)",
                          i + 1, len(waypoints), name, x, y)

            if not self.send_goal(x, y, yaw, timeout=90.0):
                rospy.logwarn("[ZoneNav] Failed to reach [%s], skipping", name)
                continue

            if self.stop_requested:
                break

            # Clear candidates so this room's scan is isolated
            self.pub_det_reset.publish(Empty())
            rospy.sleep(0.5)

            rospy.loginfo("[ZoneNav] Rotating to scan room at [%s]...", name)
            found = self.rotate_and_scan(target, duration=16.0, angular_vel=0.4)

            if found:
                enter_x = x + enter_forward_dist * math.cos(yaw)
                enter_y = y + enter_forward_dist * math.sin(yaw)
                rospy.loginfo("[ZoneNav] Entering room [%s] -> (%.2f, %.2f)",
                              name, enter_x, enter_y)
                self.send_goal(enter_x, enter_y, yaw, timeout=45.0)
                rospy.loginfo("[ZoneNav] Arrived at target box #%s. Mission complete.",
                              str(target))
                return
            else:
                rospy.loginfo("[ZoneNav] [%s] is not the target room (visible: %s), "
                              "moving to next", name, str(self.detected_counts))

        rospy.logwarn("[ZoneNav] Upper search finished without finding target #%s",
                      str(target))

    # =========================================================

    def _run_zone(self, cmd):
        """Execute zone navigation in a separate thread."""
        try:
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
