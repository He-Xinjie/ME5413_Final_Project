#!/usr/bin/env python3

import threading
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs


class DepthClickTester:
    def __init__(self):
        rospy.init_node("depth_click_test")

        self.rgb_topic = rospy.get_param("~rgb_topic", "/front_depth/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/front_depth/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/front_depth/camera_info")

        self.camera_frame = rospy.get_param("~camera_frame", "depth_camera_link")
        self.target_frame = rospy.get_param("~target_frame", "map")

        self.use_manual_intrinsics = rospy.get_param("~use_manual_intrinsics", True)
        self.manual_fx = rospy.get_param("~manual_fx", 554.26)
        self.manual_fy = rospy.get_param("~manual_fy", 554.26)
        self.manual_cx = rospy.get_param("~manual_cx", 320.0)
        self.manual_cy = rospy.get_param("~manual_cy", 240.0)

        self.bridge = CvBridge()

        self.rgb_img = None
        self.depth_img = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.lock = threading.Lock()
        self.window_name = "RGB Click Test"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        rospy.loginfo("Subscribed topics:")
        rospy.loginfo("  RGB         : %s", self.rgb_topic)
        rospy.loginfo("  Depth       : %s", self.depth_topic)
        rospy.loginfo("  Camera Info : %s", self.camera_info_topic)
        rospy.loginfo("  Camera frame: %s", self.camera_frame)
        rospy.loginfo("  Target frame: %s", self.target_frame)
        rospy.loginfo("Click on the RGB image to inspect depth.")

    def camera_info_callback(self, msg):
        with self.lock:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.rgb_img = img
        except CvBridgeError as e:
            rospy.logerr("RGB error: %s", str(e))

    def depth_callback(self, msg):
        try:
            if msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            elif msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            with self.lock:
                self.depth_img = depth
        except CvBridgeError as e:
            rospy.logerr("Depth error: %s", str(e))

    def get_depth_window_median(self, u, v, radius=2):
        if self.depth_img is None:
            return None

        h, w = self.depth_img.shape[:2]
        u0 = max(0, u - radius)
        u1 = min(w, u + radius + 1)
        v0 = max(0, v - radius)
        v1 = min(h, v + radius + 1)

        patch = self.depth_img[v0:v1, u0:u1]

        if patch.dtype == np.float32 or patch.dtype == np.float64:
            valid = patch[np.isfinite(patch)]
            valid = valid[valid > 0.0]
        else:
            valid = patch[patch > 0].astype(np.float32) / 1000.0

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    def pixel_to_camera_3d(self, u, v, z):
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        Z = z
        return X, Y, Z

    def camera_point_to_map(self, X, Y, Z):
        pt = PointStamped()
        pt.header.stamp = rospy.Time(0)
        pt.header.frame_id = self.camera_frame
        pt.point.x = X
        pt.point.y = Y
        pt.point.z = Z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            pt_map = tf2_geometry_msgs.do_transform_point(pt, transform)
            return pt_map
        except Exception as e:
            rospy.logwarn("TF transform failed: %s", str(e))
            return None

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        with self.lock:
            if self.rgb_img is None:
                rospy.logwarn("RGB not ready")
                return

            if self.depth_img is None:
                rospy.logwarn("Depth not ready")
                return

            if self.fx is None:
                if self.use_manual_intrinsics:
                    self.fx = self.manual_fx
                    self.fy = self.manual_fy
                    self.cx = self.manual_cx
                    self.cy = self.manual_cy
                    rospy.logwarn("Using manual intrinsics")
                else:
                    rospy.logwarn("Camera info not ready")
                    return

            z = self.get_depth_window_median(x, y)
            if z is None:
                rospy.logwarn("No depth at (%d, %d)", x, y)
                return

            X, Y, Z = self.pixel_to_camera_3d(x, y, z)

        rospy.loginfo("Pixel: (%d, %d)", x, y)
        rospy.loginfo("Depth: %.3f m", z)
        rospy.loginfo("Camera 3D: X=%.3f Y=%.3f Z=%.3f", X, Y, Z)

        pt_map = self.camera_point_to_map(X, Y, Z)
        if pt_map is not None:
            rospy.loginfo(
                "Map 3D: x=%.3f y=%.3f z=%.3f",
                pt_map.point.x,
                pt_map.point.y,
                pt_map.point.z,
            )

    def spin(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            with self.lock:
                img = self.rgb_img.copy() if self.rgb_img is not None else None

            if img is not None:
                cv2.putText(
                    img,
                    "Click to get depth + map position",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(self.window_name, img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = DepthClickTester()
    node.spin()
