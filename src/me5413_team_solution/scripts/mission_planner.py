#!/usr/bin/env python3
"""
Mission Planner - Waypoint Navigation State Machine for ME5413 Final Project

States:
  IDLE          -> Wait for start command
  EXPLORE_LOWER -> Navigate through lower level to observe boxes
  UNBLOCK_EXIT  -> Publish /cmd_unblock and rush to exit
  CLIMB_RAMP    -> Navigate up the ramp to upper level
  CORRIDOR_NAV  -> Navigate through corridor, detect which gap is open
  ENTER_ROOM    -> Go through open gap, avoid moving obstacle, reach target room
  DONE          -> Mission complete
"""

import rospy
import actionlib
import math
from enum import Enum
import dynamic_reconfigure.client

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus


class State(Enum):
    IDLE = 0
    EXPLORE_LOWER = 1
    UNBLOCK_EXIT = 2
    CLIMB_RAMP = 3
    CORRIDOR_NAV = 4
    ENTER_ROOM = 5
    DONE = 6


class MissionPlanner:
    def __init__(self):
        rospy.init_node('mission_planner', anonymous=False)

        # Move base action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("[MissionPlanner] Waiting for move_base action server...")
        self.move_base_client.wait_for_server(rospy.Duration(30.0))
        rospy.loginfo("[MissionPlanner] move_base action server connected!")

        # Publishers
        self.pub_unblock = rospy.Publisher('/cmd_unblock', Bool, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # State
        self.state = State.IDLE
        self.current_waypoint_idx = 0
        self.target_room_id = None  # Will be set by box counting result

        # Auto start parameter
        self.auto_start = rospy.get_param('~auto_start', False)

        # Subscriber for manual start trigger
        self.sub_start = rospy.Subscriber(
            '/mission_planner/start', Bool, self.start_callback
        )

        # Subscriber for box count result (from perception node)
        # Expected message: the box number (1-9) with minimum count
        self.sub_box_result = rospy.Subscriber(
            '/mission_planner/min_box_id', String, self.box_result_callback
        )

        # ============================================================
        # WAYPOINTS DEFINITION
        # All coordinates are in the MAP frame.
        # These need to be calibrated by running the simulation
        # and recording positions using RViz "Publish Point" tool.
        #
        # Format: (x, y, yaw) in map frame
        # ============================================================

        # --- Lower level: patrol waypoints to observe boxes ---
        # Calibrated from RViz Publish Point (map frame)
        # Zigzag pattern covering the entire box area
        self.waypoints_explore = [
            # Row 1: south edge, heading right
            (4.10,  -0.58,  0.0),
            (7.46,  -0.47,  0.0),
            (10.80, -0.33,  0.0),
            # Turn north
            (10.83,  6.79,  1.57),
            # Row 2: middle, heading left
            (7.69,   6.98,  3.14),
            (3.97,   6.79,  3.14),
            # Turn north
            (3.80,  13.18,  1.57),
            # Row 3: heading right
            (7.52,  13.14,  0.0),
            (11.11, 13.01,  0.0),
            (15.92, 13.96,  0.0),
            (18.50, 13.69,  0.0),
            (21.23, 14.19,  0.0),
            # Turn south
            (20.89,  8.66, -1.57),
            # Row 4: heading left
            (16.71,  8.49,  3.14),
            (13.86,  8.33,  3.14),
            # Turn south
            (14.00,  4.13, -1.57),
            # Row 5: heading right
            (18.22,  4.39,  0.0),
            (21.12,  4.29,  0.0),
            # Turn south
            (20.68, -0.79, -1.57),
            # Row 6: heading left
            (17.28, -0.41,  3.14),
            (12.36,  1.32,  3.14),
        ]

        # --- Waypoints to exit lower level after unblocking ---
        # Cone barrier is at world (-15, -10)
        self.waypoints_exit = [
            (-15.0, -9.0, -1.57),   # Approach exit
            (-15.0, -11.0, -1.57),  # Pass through exit
        ]

        # --- Ramp waypoints ---
        # Ramp is around world (6.5~13.5, -11~-13) going from z=0 to z=2.5
        self.waypoints_ramp = [
            (-5.0, -11.0, 0.0),     # Approach ramp area
            (0.0,  -11.0, 0.0),     # Continue toward ramp
            (6.0,  -12.0, 0.0),     # Ramp start
            (10.0, -11.5, 0.0),     # On the ramp
            (13.0, -11.0, 0.0),     # Top of ramp
        ]

        # --- Corridor waypoints (upper level) ---
        # Corridor along x~12, two gaps at y~5 and y~-5
        self.waypoints_corridor = [
            (14.0, -10.0, 1.57),    # Enter corridor area
            (12.0, -8.0, 1.57),     # Move up corridor
        ]

        # --- Room entry waypoints ---
        # After detecting which gap is open (y~5 or y~-5)
        # Room waypoints depend on which gap + which target room
        # These are templates; actual target selected at runtime

        # Through gap at y~-5 (south corridor)
        self.waypoints_gap_south = [
            (12.0, -5.0, 3.14),     # Approach south gap
            (8.0,  -5.0, 3.14),     # Through gap
        ]

        # Through gap at y~5 (north corridor)
        self.waypoints_gap_north = [
            (12.0,  5.0, 3.14),     # Approach north gap
            (8.0,   5.0, 3.14),     # Through gap
        ]

        # Target room positions (upper level, near destination boxes)
        # Destination boxes at world x=2.5, y spread from -10 to 10, z=2.6
        # 4 box types, spaced along y-axis
        # Room 1 (south): y ~ -7.5
        # Room 2 (south-mid): y ~ -2.5
        # Room 3 (north-mid): y ~ 2.5
        # Room 4 (north): y ~ 7.5
        self.room_positions = {
            1: (3.0, -7.5, 3.14),
            2: (3.0, -2.5, 3.14),
            3: (3.0,  2.5, 3.14),
            4: (3.0,  7.5, 3.14),
        }

        rospy.loginfo("[MissionPlanner] Initialized. Waiting for start command...")
        rospy.loginfo("[MissionPlanner] Publish 'true' to /mission_planner/start to begin")

        if self.auto_start:
            rospy.sleep(3.0)
            rospy.loginfo("[MissionPlanner] Auto-starting mission...")
            self.state = State.EXPLORE_LOWER
            self.run()

    def start_callback(self, msg):
        if msg.data and self.state == State.IDLE:
            rospy.loginfo("[MissionPlanner] Start command received!")
            self.state = State.EXPLORE_LOWER
            self.run()

    def box_result_callback(self, msg):
        """Receive the box ID with minimum occurrence count."""
        try:
            self.target_room_id = int(msg.data)
            rospy.loginfo(
                "[MissionPlanner] Target room set to box_%d (minimum count)",
                self.target_room_id
            )
        except ValueError:
            rospy.logwarn("[MissionPlanner] Invalid box result: %s", msg.data)

    def create_goal(self, x, y, yaw):
        """Create a MoveBaseGoal from x, y, yaw in map frame."""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)

        return goal

    def send_goal_and_wait(self, x, y, yaw, timeout=60.0):
        """Send a navigation goal and wait for result."""
        goal = self.create_goal(x, y, yaw)
        rospy.loginfo(
            "[MissionPlanner] Navigating to (%.1f, %.1f, yaw=%.2f)...", x, y, yaw
        )

        self.move_base_client.send_goal(goal)
        finished = self.move_base_client.wait_for_result(
            rospy.Duration(timeout)
        )

        if finished:
            state = self.move_base_client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("[MissionPlanner] Reached goal (%.1f, %.1f)", x, y)
                return True
            else:
                rospy.logwarn(
                    "[MissionPlanner] Goal (%.1f, %.1f) failed with status %d",
                    x, y, state
                )
                return False
        else:
            rospy.logwarn(
                "[MissionPlanner] Goal (%.1f, %.1f) timed out", x, y
            )
            self.move_base_client.cancel_goal()
            return False

    def navigate_waypoints(self, waypoints, label="", timeout=60.0):
        """Navigate through a list of waypoints sequentially."""
        for i, (x, y, yaw) in enumerate(waypoints):
            if rospy.is_shutdown():
                return False
            rospy.loginfo(
                "[MissionPlanner] %s waypoint %d/%d",
                label, i + 1, len(waypoints)
            )
            success = self.send_goal_and_wait(x, y, yaw, timeout)
            if not success:
                rospy.logwarn(
                    "[MissionPlanner] %s waypoint %d failed, continuing...",
                    label, i + 1
                )
                # Don't abort the whole mission on a single waypoint failure
                rospy.sleep(1.0)
        return True

    def stop_robot(self):
        """Send zero velocity to stop the robot."""
        stop_msg = Twist()
        self.pub_cmd_vel.publish(stop_msg)

    def unblock_and_exit(self):
        """Publish unblock command and immediately head for the exit."""
        rospy.loginfo("[MissionPlanner] Sending unblock command!")
        msg = Bool()
        msg.data = True
        self.pub_unblock.publish(msg)
        rospy.sleep(0.5)
        # Publish again to ensure delivery
        self.pub_unblock.publish(msg)

        rospy.loginfo("[MissionPlanner] Rushing to exit (10s window)!")
        # Navigate to exit with short timeout - we only have 10 seconds
        for x, y, yaw in self.waypoints_exit:
            if rospy.is_shutdown():
                return False
            self.send_goal_and_wait(x, y, yaw, timeout=15.0)
        return True

    def run(self):
        """Main state machine loop."""
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.state == State.IDLE:
                rate.sleep()
                continue

            elif self.state == State.EXPLORE_LOWER:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] STATE: EXPLORE_LOWER")
                rospy.loginfo("=" * 50)

                self.navigate_waypoints(
                    self.waypoints_explore, "Explore", timeout=45.0
                )

                # Wait a moment for perception to finish counting
                rospy.loginfo(
                    "[MissionPlanner] Exploration done. "
                    "Waiting for box count result..."
                )

                # If no perception node is running, use a default target
                wait_start = rospy.Time.now()
                while (
                    self.target_room_id is None
                    and (rospy.Time.now() - wait_start).to_sec() < 10.0
                    and not rospy.is_shutdown()
                ):
                    rate.sleep()

                if self.target_room_id is None:
                    rospy.logwarn(
                        "[MissionPlanner] No box count received, "
                        "defaulting to room 1"
                    )
                    self.target_room_id = 1

                self.state = State.UNBLOCK_EXIT

            elif self.state == State.UNBLOCK_EXIT:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] STATE: UNBLOCK_EXIT")
                rospy.loginfo("=" * 50)

                self.unblock_and_exit()
                self.state = State.CLIMB_RAMP

            elif self.state == State.CLIMB_RAMP:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] STATE: CLIMB_RAMP")
                rospy.loginfo("=" * 50)

                # Reduce speed/acceleration for safe ramp traversal
                try:
                    teb_client = dynamic_reconfigure.client.Client(
                        '/move_base/TebLocalPlannerROS', timeout=5)
                    teb_client.update_configuration({
                        'max_vel_x': 0.3,
                        'max_vel_theta': 0.5,
                        'acc_lim_x': 1.0,
                        'acc_lim_theta': 1.5,
                    })
                    rospy.loginfo("[MissionPlanner] Ramp mode: reduced speed")
                except Exception:
                    rospy.logwarn("[MissionPlanner] Failed to set ramp speed")

                self.navigate_waypoints(
                    self.waypoints_ramp, "Ramp", timeout=45.0
                )

                # Restore normal speed after ramp
                try:
                    teb_client.update_configuration({
                        'max_vel_x': 0.5,
                        'max_vel_theta': 1.5,
                        'acc_lim_x': 2.5,
                        'acc_lim_theta': 3.2,
                    })
                    rospy.loginfo("[MissionPlanner] Normal speed restored")
                except Exception:
                    pass

                self.state = State.CORRIDOR_NAV

            elif self.state == State.CORRIDOR_NAV:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] STATE: CORRIDOR_NAV")
                rospy.loginfo("=" * 50)

                self.navigate_waypoints(
                    self.waypoints_corridor, "Corridor", timeout=45.0
                )

                # TODO: Add cone detection here to choose gap
                # For now, try south gap first; if blocked, try north
                # The cone detection node should publish to
                # /mission_planner/open_gap with "north" or "south"
                rospy.loginfo(
                    "[MissionPlanner] Attempting south corridor gap..."
                )
                self.state = State.ENTER_ROOM

            elif self.state == State.ENTER_ROOM:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] STATE: ENTER_ROOM")
                rospy.loginfo("=" * 50)

                # TODO: Choose gap based on cone detection
                # Default: try south gap
                gap_waypoints = self.waypoints_gap_south

                self.navigate_waypoints(
                    gap_waypoints, "Gap", timeout=45.0
                )

                # Navigate to target room
                if self.target_room_id in self.room_positions:
                    target = self.room_positions[self.target_room_id]
                    rospy.loginfo(
                        "[MissionPlanner] Heading to target room %d",
                        self.target_room_id
                    )
                    self.send_goal_and_wait(
                        target[0], target[1], target[2], timeout=60.0
                    )
                else:
                    rospy.logwarn(
                        "[MissionPlanner] Unknown room id %s, going to room 1",
                        self.target_room_id
                    )
                    target = self.room_positions[1]
                    self.send_goal_and_wait(
                        target[0], target[1], target[2], timeout=60.0
                    )

                self.state = State.DONE

            elif self.state == State.DONE:
                rospy.loginfo("=" * 50)
                rospy.loginfo("[MissionPlanner] MISSION COMPLETE!")
                rospy.loginfo("=" * 50)
                self.stop_robot()
                break

            rate.sleep()


if __name__ == '__main__':
    try:
        planner = MissionPlanner()
        if not planner.auto_start:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
