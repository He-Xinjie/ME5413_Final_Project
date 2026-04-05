#!/usr/bin/env python3

import os
import math
import yaml
import rospy
import actionlib

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Bool


class WaypointPatrol:
    def __init__(self):
        rospy.init_node("waypoint_patrol")

        # YAML waypoint 文件
        self.waypoint_file = rospy.get_param(
            "~waypoint_file",
            os.path.expanduser(
                "~/ME5413_Final_Project/src/me5413_team_solution/config/waypoints.yaml"
            ),
        )

        # 巡航行为参数
        self.loop_patrol = rospy.get_param("~loop_patrol", False)
        self.pause_at_waypoint = rospy.get_param("~pause_at_waypoint", 3.0)
        self.retry_per_waypoint = rospy.get_param("~retry_per_waypoint", 2)
        self.goal_timeout = rospy.get_param("~goal_timeout", 120.0)

        # 主动扫描校正参数
        self.enable_relocalization_scan = rospy.get_param("~enable_relocalization_scan", True)
        self.scan_angular_speed = rospy.get_param("~scan_angular_speed", 0.35)  # rad/s
        self.scan_angle_deg = rospy.get_param("~scan_angle_deg", 20.0)          # 单边角度
        self.scan_settle_time = rospy.get_param("~scan_settle_time", 1.0)       # 扫描后等待

        # /initialpose 自动重定位参数
        self.enable_initialpose_relocalization = rospy.get_param("~enable_initialpose_relocalization", True)
        self.relocalize_pause = rospy.get_param("~relocalize_pause", 1.0)

        # 协方差（你可以后面再微调）
        self.initialpose_cov_x = rospy.get_param("~initialpose_cov_x", 0.05)
        self.initialpose_cov_y = rospy.get_param("~initialpose_cov_y", 0.05)
        self.initialpose_cov_yaw = rospy.get_param("~initialpose_cov_yaw", 0.10)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self.detect_enable_pub = rospy.Publisher("/box_counter/enable", Bool, queue_size=1, latch=True)

        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        rospy.loginfo("Loading waypoints from: %s", self.waypoint_file)
        self.waypoints = self.load_waypoints(self.waypoint_file)
        rospy.loginfo("Loaded %d waypoints.", len(self.waypoints))

        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base.")
        self.detect_enable_pub.publish(Bool(data=False))
        rospy.loginfo("Box counting disabled by default.")

    @staticmethod
    def load_waypoints(file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        if "waypoints" not in data or not isinstance(data["waypoints"], list):
            raise ValueError("YAML must contain a list under key 'waypoints'.")

        waypoints = []
        for i, wp in enumerate(data["waypoints"]):
            for key in ["x", "y", "z", "w"]:
                if key not in wp:
                    raise ValueError(f"Waypoint {i+1} missing key: {key}")
            waypoints.append((wp["x"], wp["y"], wp["z"], wp["w"]))
        return waypoints

    @staticmethod
    def create_goal(x, y, z, w):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w

        return goal

    def stop_robot(self, duration=0.2):
        twist = Twist()
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(10)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            rate.sleep()

    def rotate_for_duration(self, angular_z, duration):
        twist = Twist()
        twist.angular.z = angular_z
        rate = rospy.Rate(20)
        end_time = rospy.Time.now() + rospy.Duration(duration)

        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_robot(0.3)

    def relocalization_scan(self):
        if not self.enable_relocalization_scan:
            return

        angle_rad = math.radians(self.scan_angle_deg)
        duration_one_side = angle_rad / max(abs(self.scan_angular_speed), 1e-3)
    
        # 开启识别
        self.set_box_counter_enabled(True)
        rospy.sleep(0.5)

        rospy.loginfo("Detection scan: center settle")
        self.stop_robot(0.5)
        rospy.sleep(self.scan_settle_time)

        rospy.loginfo("Detection scan: left %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(+self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        rospy.loginfo("Detection scan: right %.1f deg", self.scan_angle_deg * 2.0)
        self.rotate_for_duration(-self.scan_angular_speed, duration_one_side * 2.0)
        rospy.sleep(self.scan_settle_time)

        rospy.loginfo("Detection scan: return center %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(+self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        # 关闭识别
        self.set_box_counter_enabled(False)
        rospy.sleep(0.5)

    def publish_initialpose(self, x, y, z, w):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w

        cov = [0.0] * 36
        cov[0] = self.initialpose_cov_x     # x
        cov[7] = self.initialpose_cov_y     # y
        cov[35] = self.initialpose_cov_yaw  # yaw
        msg.pose.covariance = cov

        self.initpose_pub.publish(msg)
        rospy.loginfo(
            "Published /initialpose at (%.3f, %.3f), cov=(%.3f, %.3f, %.3f)",
            x, y,
            self.initialpose_cov_x,
            self.initialpose_cov_y,
            self.initialpose_cov_yaw
        )

    def set_box_counter_enabled(self, enabled):
        self.detect_enable_pub.publish(Bool(data=enabled))
        rospy.loginfo("Box counter enabled: %s", enabled)

    def goto_waypoint(self, wp_idx, waypoint):
        x, y, z, w = waypoint
        goal = self.create_goal(x, y, z, w)

        for attempt in range(1, self.retry_per_waypoint + 2):
            if rospy.is_shutdown():
                return False

            rospy.loginfo(
                "Waypoint %d | attempt %d | goal=(%.3f, %.3f)",
                wp_idx + 1,
                attempt,
                x,
                y
            )

            self.client.send_goal(goal)
            finished = self.client.wait_for_result(rospy.Duration(self.goal_timeout))

            if not finished:
                rospy.logwarn("Waypoint %d timed out. Cancelling goal...", wp_idx + 1)
                self.client.cancel_goal()
                self.stop_robot(0.5)
            else:
                state = self.client.get_state()

                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Waypoint %d reached.", wp_idx + 1)

                    # 先停住
                    self.stop_robot(0.5)

                    # 到点后停顿
                    rospy.sleep(self.pause_at_waypoint)

                    # 原地左右扫描，帮助 AMCL 收敛
                    self.relocalization_scan()

                    # 在每个 waypoint 都发布 /initialpose
                    if self.enable_initialpose_relocalization:
                        rospy.loginfo("Publishing /initialpose at waypoint %d", wp_idx + 1)
                        rospy.sleep(self.relocalize_pause)
                        self.publish_initialpose(x, y, z, w)
                        rospy.sleep(1.0)

                    return True
                else:
                    rospy.logwarn(
                        "Waypoint %d failed with state=%d. Retrying...",
                        wp_idx + 1,
                        state
                    )
                    self.stop_robot(0.5)

            rospy.sleep(1.0)

        rospy.logerr("Waypoint %d failed after all retries. Skipping.", wp_idx + 1)
        return False

    def run(self):
        if not self.waypoints:
            rospy.logerr("No waypoints loaded.")
            return

        lap = 1
        while not rospy.is_shutdown():
            rospy.loginfo("Starting patrol lap %d", lap)

            for i, wp in enumerate(self.waypoints):
                success = self.goto_waypoint(i, wp)
                if not success:
                    rospy.logwarn("Skipping waypoint %d and continuing patrol.", i + 1)

            rospy.loginfo("Completed patrol lap %d", lap)

            if not self.loop_patrol:
                break

            lap += 1
            rospy.sleep(2.0)


if __name__ == "__main__":
    try:
        node = WaypointPatrol()
        node.run()
    except Exception as e:
        rospy.logerr("Waypoint patrol node crashed: %s", str(e))
