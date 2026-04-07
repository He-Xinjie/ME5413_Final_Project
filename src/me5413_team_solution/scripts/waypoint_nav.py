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

        # =========================
        # 巡航参数
        # =========================
        self.loop_patrol = rospy.get_param("~loop_patrol", False)
        self.pause_at_waypoint = rospy.get_param("~pause_at_waypoint", 0.5)
        self.retry_per_waypoint = rospy.get_param("~retry_per_waypoint", 1)
        self.goal_timeout = rospy.get_param("~goal_timeout", 40.0)

        # =========================
        # 扫描识别参数
        # 目标：每个点左右转，并持续识别更久
        # =========================
        self.enable_relocalization_scan = rospy.get_param("~enable_relocalization_scan", True)

        # 转动更慢一些，给 box_counter 更多帧
        self.scan_angular_speed = rospy.get_param("~scan_angular_speed", 0.25)   # rad/s

        # 左右扫描角度更大
        self.scan_angle_deg = rospy.get_param("~scan_angle_deg", 45.0)          # 单边角度

        # 每次转到位后停一下，让识别持续进行
        self.scan_settle_time = rospy.get_param("~scan_settle_time", 0.8)

        # 额外的中心观察时间：开始、左、右、回中都可观察更久
        self.center_observe_time = rospy.get_param("~center_observe_time", 0.8)
        self.side_observe_time = rospy.get_param("~side_observe_time", 1.0)

        # 如果为空，则默认每个 waypoint 都扫描
        self.scan_waypoint_indices = rospy.get_param("~scan_waypoint_indices", [])

        # =========================
        # /initialpose 自动重定位参数
        # =========================
        self.enable_initialpose_relocalization = rospy.get_param("~enable_initialpose_relocalization", True)
        self.relocalize_pause = rospy.get_param("~relocalize_pause", 0.5)

        # 协方差
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

        rospy.loginfo(
            "Waypoint mode: pause_at_waypoint=%.2f, retry_per_waypoint=%d, goal_timeout=%.1f",
            self.pause_at_waypoint,
            self.retry_per_waypoint,
            self.goal_timeout
        )
        rospy.loginfo(
            "Relocalization scan enabled: %s | initialpose relocalization enabled: %s",
            self.enable_relocalization_scan,
            self.enable_initialpose_relocalization
        )

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
        rate = rospy.Rate(15)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            rate.sleep()

    def observe_for_duration(self, duration):
        """
        机器人静止观察，让 box_counter 在这一段时间持续识别。
        """
        self.stop_robot(0.1)
        rospy.sleep(duration)

    def rotate_for_duration(self, angular_z, duration):
        twist = Twist()
        twist.angular.z = angular_z
        rate = rospy.Rate(20)
        end_time = rospy.Time.now() + rospy.Duration(duration)

        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_robot(0.2)

    def should_scan_at_waypoint(self, wp_idx):
        if not self.enable_relocalization_scan:
            return False

        # 如果没有指定 waypoint 列表，则默认每个点都扫描
        if not self.scan_waypoint_indices:
            return True

        return wp_idx in self.scan_waypoint_indices

    def relocalization_scan(self):
        """
        扫描流程：
        1. 中间观察
        2. 左转45°
        3. 左侧观察
        4. 回中
        5. 中间观察
        6. 右转45°
        7. 右侧观察
        8. 回中
        9. 中间再观察
        全程保持检测开启
        """
        angle_rad = math.radians(self.scan_angle_deg)
        duration_one_side = angle_rad / max(abs(self.scan_angular_speed), 1e-3)

        # 开启识别
        self.set_box_counter_enabled(True)
        rospy.sleep(0.3)

        # 0) 初始中间观察
        rospy.loginfo("Detection scan: center observe")
        self.observe_for_duration(self.center_observe_time)

        # 1) 左转 45°
        rospy.loginfo("Detection scan: turn left %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(+self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        # 2) 左侧观察
        rospy.loginfo("Detection scan: observe left")
        self.observe_for_duration(self.side_observe_time)

        # 3) 回中
        rospy.loginfo("Detection scan: return to center from left %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(-self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        # 4) 中间再观察
        rospy.loginfo("Detection scan: center observe after left")
        self.observe_for_duration(self.center_observe_time)

        # 5) 右转 45°
        rospy.loginfo("Detection scan: turn right %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(-self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        # 6) 右侧观察
        rospy.loginfo("Detection scan: observe right")
        self.observe_for_duration(self.side_observe_time)

        # 7) 回中
        rospy.loginfo("Detection scan: return to center from right %.1f deg", self.scan_angle_deg)
        self.rotate_for_duration(+self.scan_angular_speed, duration_one_side)
        rospy.sleep(self.scan_settle_time)

        # 8) 最终中间观察
        rospy.loginfo("Detection scan: final center observe")
        self.observe_for_duration(self.center_observe_time)

        # 关闭识别
        self.set_box_counter_enabled(False)
        rospy.sleep(0.3)

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
        cov[0] = self.initialpose_cov_x
        cov[7] = self.initialpose_cov_y
        cov[35] = self.initialpose_cov_yaw
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
                self.stop_robot(0.2)
            else:
                state = self.client.get_state()

                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Waypoint %d reached.", wp_idx + 1)

                    # 轻微停稳
                    self.stop_robot(0.2)

                    # 到点短暂停顿
                    rospy.sleep(self.pause_at_waypoint)

                    # 每个点都可以扫描
                    if self.should_scan_at_waypoint(wp_idx):
                        self.relocalization_scan()

                    # 扫描结束后，再做 initialpose
                    if self.enable_initialpose_relocalization:
                        rospy.loginfo("Publishing /initialpose at waypoint %d", wp_idx + 1)
                        rospy.sleep(self.relocalize_pause)
                        self.publish_initialpose(x, y, z, w)
                        rospy.sleep(0.5)

                    return True
                else:
                    rospy.logwarn(
                        "Waypoint %d failed with state=%d. Retrying...",
                        wp_idx + 1,
                        state
                    )
                    self.stop_robot(0.2)

            rospy.sleep(0.5)

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
            rospy.sleep(1.0)


if __name__ == "__main__":
    try:
        node = WaypointPatrol()
        node.run()
    except Exception as e:
        rospy.logerr("Waypoint patrol node crashed: %s", str(e))
