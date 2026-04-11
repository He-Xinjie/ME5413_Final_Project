#!/usr/bin/env python3
"""
Box Detector Node - Detects numbered boxes (1-9) using YOLO11 + front camera.

Uses YOLO11s for digit detection instead of template matching.
Spatial deduplication prevents double-counting the same box.
"""

import rospy
import cv2
import numpy as np
import os
import math
import tf2_ros
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class BoxDetector:
    def __init__(self):
        rospy.init_node('box_detector', anonymous=False)

        self.bridge = CvBridge()

        # TF for robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Box count: {number: count}
        self.box_counts = {}
        # Spatial deduplication: {number: [(x1,y1), (x2,y2), ...]}
        self.detected_positions = {}
        self.dedup_radius = 4.0
        # Cooldown: after counting a new box, ignore that number briefly
        self.detection_cooldown = {}
        self.cooldown_frames = 10

        # YOLO parameters
        self.yolo_conf = rospy.get_param('~yolo_conf', 0.5)
        self.yolo_iou = rospy.get_param('~yolo_iou', 0.45)

        # Load YOLO model
        model_path = rospy.get_param('~model_path', '')
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'Final.pt'
            )
        rospy.loginfo("[BoxDetector] Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)
        rospy.loginfo("[BoxDetector] YOLO model loaded. Classes: %s", self.model.names)

        # Publishers
        self.pub_min_box = rospy.Publisher(
            '/mission_planner/min_box_id', String, queue_size=1, latch=True)
        self.pub_viz = rospy.Publisher(
            '/box_detector/image', Image, queue_size=1)
        self.pub_counts = rospy.Publisher(
            '/box_detector/counts', String, queue_size=1)

        # Subscriber
        self.sub_image = rospy.Subscriber(
            '/front/image_raw', Image, self.image_callback,
            queue_size=1, buff_size=2**24)

        self.robot_x = 0.0
        self.robot_y = 0.0
        rospy.loginfo("[BoxDetector] Initialized.")

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(0.5))
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
        except Exception:
            pass
        return self.robot_x, self.robot_y

    def _is_new_box(self, num, rx, ry):
        if num not in self.detected_positions:
            return True
        for (px, py) in self.detected_positions[num]:
            if math.sqrt((rx - px)**2 + (ry - py)**2) < self.dedup_radius:
                return False
        return True

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logwarn("[BoxDetector] cv_bridge error: %s", str(e))
            return

        # Decrease cooldowns
        for num in list(self.detection_cooldown.keys()):
            self.detection_cooldown[num] -= 1
            if self.detection_cooldown[num] <= 0:
                del self.detection_cooldown[num]

        # --- YOLO inference ---
        results = self.model.predict(
            cv_image,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=640,
            verbose=False
        )

        viz_image = cv_image.copy()
        rx, ry = self._get_robot_pose()

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                digit = int(self.model.names[cls_id])

                color = (0, 255, 0)

                if digit not in self.detection_cooldown:
                    if self._is_new_box(digit, rx, ry):
                        self.box_counts[digit] = self.box_counts.get(digit, 0) + 1
                        if digit not in self.detected_positions:
                            self.detected_positions[digit] = []
                        self.detected_positions[digit].append((rx, ry))
                        self.detection_cooldown[digit] = self.cooldown_frames
                        color = (0, 0, 255)  # red for new detection
                        rospy.loginfo(
                            "[BoxDetector] NEW box #%d at (%.1f,%.1f) conf=%.2f Counts: %s",
                            digit, rx, ry, conf, str(self.box_counts))
                    else:
                        self.detection_cooldown[digit] = self.cooldown_frames

                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_image, "#%d %.2f" % (digit, conf),
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw counts overlay
        y_off = 25
        cv2.putText(viz_image, "YOLO Counts:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_off += 25
        for num in sorted(self.box_counts.keys()):
            cv2.putText(viz_image, "  #%d: %d" % (num, self.box_counts[num]),
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_off += 22

        # Publish
        try:
            self.pub_viz.publish(self.bridge.cv2_to_imgmsg(viz_image, 'bgr8'))
        except Exception:
            pass

        self.pub_counts.publish(String(data=str(self.box_counts)))

        if self.box_counts:
            min_box = min(self.box_counts, key=self.box_counts.get)
            self.pub_min_box.publish(String(data=str(min_box)))


if __name__ == '__main__':
    try:
        detector = BoxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
