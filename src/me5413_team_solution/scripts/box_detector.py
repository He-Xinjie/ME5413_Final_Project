#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs


class BoxDetector:
    def __init__(self):
        rospy.init_node("box_detector")

        self.rgb_topic = "/front_depth/image_raw"
        self.depth_topic = "/front_depth/depth/image_raw"

        self.bridge = CvBridge()
        self.rgb_img = None
        self.depth_img = None

        # 手动内参
        self.fx = 554.26
        self.fy = 554.26
        self.cx = 320.0
        self.cy = 240.0

        self.camera_frame = "depth_camera_link"
        self.target_frame = "map"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)

        self.window = "Box Detector"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def rgb_callback(self, msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        if msg.encoding == "32FC1":
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        else:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def get_depth_median(self, u, v, radius=2):
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

    def pixel_to_3d(self, u, v, z):
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return X, Y, z

    def to_map(self, X, Y, Z):
        pt = PointStamped()
        pt.header.frame_id = self.camera_frame
        pt.header.stamp = rospy.Time(0)
        pt.point.x = X
        pt.point.y = Y
        pt.point.z = Z

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            return tf2_geometry_msgs.do_transform_point(pt, tf)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF transform failed: %s", str(e))
            return None

    def detect_digits(self, img):
        """
        检测黑色数字区域，而不是检测整块灰色箱子。
        返回若干 bounding boxes: (x, y, w, h)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 找黑色数字
        mask = cv2.inRange(gray, 0, 60)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        h_img, w_img = gray.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # 排除太大或太细长的区域
            if w > 0.5 * w_img or h > 0.5 * h_img:
                continue
            if w < 10 or h < 20:
                continue

            ratio = w / float(h)
            if ratio < 0.15 or ratio > 1.2:
                continue

            boxes.append((x, y, w, h))

        return boxes, mask

    def estimate_box_center_from_digit(self, x, y, w, h):
        """
        从数字框反推箱子中心。
        对正面视角，数字通常大致在箱子正面的中间。
        先用数字框中心作为近似中心。
        """
        u = int(x + w / 2)
        v = int(y + h / 2)
        return u, v

    def spin(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.rgb_img is None:
                rate.sleep()
                continue

            img = self.rgb_img.copy()
            digit_boxes, mask = self.detect_digits(img)

            for (x, y, w, h) in digit_boxes:
                u, v = self.estimate_box_center_from_digit(x, y, w, h)

                z = self.get_depth_median(u, v)
                if z is None:
                    continue

                X, Y, Z = self.pixel_to_3d(u, v, z)
                pt_map = self.to_map(X, Y, Z)

                # 画数字框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img, (u, v), 5, (0, 0, 255), -1)

                if pt_map is not None:
                    text = f"x={pt_map.point.x:.2f}, y={pt_map.point.y:.2f}"
                    cv2.putText(
                        img,
                        text,
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    rospy.loginfo_throttle(
                        1.0,
                        "DIGIT -> map: x=%.2f, y=%.2f",
                        pt_map.point.x,
                        pt_map.point.y,
                    )

            cv2.imshow(self.window, img)
            cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    node = BoxDetector()
    node.spin()
