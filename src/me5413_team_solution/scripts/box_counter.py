#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs

from digit_classifier import DigitClassifier
from std_msgs.msg import Bool


class BoxCounter:
    def __init__(self):
        rospy.init_node("box_counter")

        self.rgb_topic = "/front_depth/image_raw"
        self.depth_topic = "/front_depth/depth/image_raw"

        self.bridge = CvBridge()
        self.rgb_img = None
        self.depth_img = None

        # Camera intrinsics
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

        self.enable_detection = False
        rospy.Subscriber("/box_counter/enable", Bool, self.enable_callback, queue_size=1)

        self.window = "Box Counter"
        self.mask_window = "digit_mask"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.mask_window, cv2.WINDOW_NORMAL)

        # Offline classifier
        self.classifier = DigitClassifier()
        rospy.loginfo("Digit classifier loaded.")

        # -------- tracking / confirmation --------
        self.tracks = []
        self.confirmed = []
        
        self.assoc_dist = 0.80
        self.confirmed_dist = 0.45
        self.confirm_votes = 6
        self.max_missed = 12
        self.score_thresh = 0.60
        self.cluster_dist_thresh = 1.2
        self.min_points_per_cluster = 3

        self.last_print_time = rospy.Time.now()

    def rgb_callback(self, msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        if msg.encoding == "32FC1":
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        else:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def enable_callback(self, msg):
        self.enable_detection = msg.data
        rospy.loginfo("box_counter enable_detection = %s", self.enable_detection)

    def get_depth_median(self, u, v, radius=5):
        if self.depth_img is None:
            return None

        h, w = self.depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

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
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            return tf2_geometry_msgs.do_transform_point(pt, tf_msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF transform failed: %s", str(e))
            return None

    def detect_digits(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold dark digits
        mask = cv2.inRange(gray, 0, 100)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        h_img, w_img = gray.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 5 or h < 10:
                continue
            if w > 0.8 * w_img or h > 0.8 * h_img:
                continue

            ratio = w / float(h)
            if ratio < 0.08 or ratio > 1.6:
                continue

            boxes.append((x, y, w, h))

        return boxes, mask

    def update_tracks(self, digit, x, y, score):
        for tr in self.tracks:
            dist = np.hypot(tr["x"] - x, tr["y"] - y)
            if dist < self.assoc_dist:
                tr["x"] = 0.7 * tr["x"] + 0.3 * x
                tr["y"] = 0.7 * tr["y"] + 0.3 * y
                tr["votes"][digit] = tr["votes"].get(digit, 0) + 1
                tr["last_score"] = score
                tr["seen"] += 1
                tr["missed"] = 0
                return

        self.tracks.append({
            "x": x,
            "y": y,
            "votes": {digit: 1},
            "seen": 1,
            "missed": 0,
            "last_score": score,
        })

    def age_tracks(self):
        keep_tracks = []
        for tr in self.tracks:
            tr["missed"] += 1
            if tr["missed"] <= self.max_missed:
                keep_tracks.append(tr)
        self.tracks = keep_tracks

    def confirm_tracks(self):
        remaining_tracks = []

        for tr in self.tracks:
            best_digit = max(tr["votes"], key=tr["votes"].get)
            best_votes = tr["votes"][best_digit]

            if best_votes >= self.confirm_votes:
                duplicated = False
                for cf in self.confirmed:
                    # 这里先不要求 digit 相同，纯按空间去重更稳
                    dist = np.hypot(cf["x"] - tr["x"], cf["y"] - tr["y"])
                    if dist < self.confirmed_dist:
                        duplicated = True
                        break

                if not duplicated:
                    self.confirmed.append({
                        "digit": best_digit,
                        "x": tr["x"],
                        "y": tr["y"],
                    })
                    rospy.loginfo(
                        "CONFIRMED BOX OBS: digit=%d at map(%.2f, %.2f)",
                        best_digit, tr["x"], tr["y"]
                    )
            else:
                remaining_tracks.append(tr)

        self.tracks = remaining_tracks

    def cluster_confirmed_boxes(self, dist_thresh=1.0, min_points_per_cluster=2):
        """
        先按空间聚类（不区分数字），再在每个簇内做数字投票。

        这样更符合：
        - 一个箱子被多次观测
        - 同一个箱子可能存在误识别
        - 四面都有数字，本质是同一个物理物体
        """
        if not self.confirmed:
            return []

        points = self.confirmed[:]
        used = [False] * len(points)
        clusters = []

        # -------- Step 1: 空间聚类 --------
        for i in range(len(points)):
            if used[i]:
                continue

            cluster_indices = [i]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(points)):
                    if used[j]:
                        continue

                    px = points[j]["x"]
                    py = points[j]["y"]

                    for idx in cluster_indices:
                        qx = points[idx]["x"]
                        qy = points[idx]["y"]

                        dist = math.hypot(px - qx, py - qy)
                        if dist <= dist_thresh:
                            cluster_indices.append(j)
                            used[j] = True
                            changed = True
                            break

            clusters.append(cluster_indices)

        # -------- Step 2: 每个簇内部数字投票 --------
        results = []
        for cluster in clusters:
            if len(cluster) < min_points_per_cluster:
                # 丢掉太孤立的小簇，减少噪声
                continue

            xs = [points[idx]["x"] for idx in cluster]
            ys = [points[idx]["y"] for idx in cluster]
            digits = [points[idx]["digit"] for idx in cluster]

            vote_count = {}
            for d in digits:
                vote_count[d] = vote_count.get(d, 0) + 1

            final_digit = max(vote_count, key=vote_count.get)

            results.append({
                "digit": final_digit,
                "x": float(sum(xs) / len(xs)),
                "y": float(sum(ys) / len(ys)),
                "count": len(cluster),
                "votes": vote_count
            })

        return results

    def get_top4_digit_counts(self):
        clustered = self.cluster_confirmed_boxes(
            dist_thresh=self.cluster_dist_thresh,
            min_points_per_cluster=self.min_points_per_cluster
        )

        raw_counts = {}
        for det in clustered:
            d = det["digit"]
            raw_counts[d] = raw_counts.get(d, 0) + 1

        # 只保留出现次数最多的4个数字
        top4 = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        top4_digits = set([d for d, _ in top4])

        final_counts = {d: c for d, c in raw_counts.items() if d in top4_digits}
        return final_counts, clustered

    def get_counts(self):
        final_counts, _ = self.get_top4_digit_counts()
        return final_counts

    def print_final_result(self):
        counts, clustered = self.get_top4_digit_counts()

        if not counts:
            rospy.logwarn("No valid detections.")
            return

        rospy.loginfo("========== FINAL RESULT ==========")
        rospy.loginfo("Digits and counts: %s", counts)

        least_digit = min(counts, key=counts.get)
        least_count = counts[least_digit]

        rospy.loginfo(
            "Least frequent digit: %d (count=%d)",
            least_digit,
            least_count
        )
        rospy.loginfo("Clustered boxes total: %d", len(clustered))
        rospy.loginfo("Raw confirmed total: %d", len(self.confirmed))
        rospy.loginfo("==================================")

    def spin(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.rgb_img is None:
                rate.sleep()
                continue

            img = self.rgb_img.copy()

            if not self.enable_detection:
                blank_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.imshow(self.window, img)
                cv2.imshow(self.mask_window, blank_mask)
                cv2.waitKey(1)
                rate.sleep()
                continue

            digit_boxes, mask = self.detect_digits(img)
            rospy.loginfo_throttle(1.0, "Detected %d digit candidates", len(digit_boxes))

            self.age_tracks()

            for (x, y, w, h) in digit_boxes:
                # 过滤太小的数字框
                if h < 25 or w < 12:
                    continue

                # 过滤极端倾斜/异常比例
                ratio = w / float(h)
                if ratio < 0.20 or ratio > 1.20:
                    continue

                roi = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                digit, score = self.classifier.classify(roi)

                rospy.loginfo_throttle(
                    1.0,
                    "CLASSIFY: pred=%s score=%.3f bbox=(%d,%d,%d,%d)",
                    str(digit), score if score is not None else -1.0, x, y, w, h
                )

                # Draw candidate box in yellow
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

                if digit is None or score is None:
                    continue

                if score < self.score_thresh:
                    continue

                # Sample depth slightly below digit center
                u = int(x + w / 2)
                v = int(y + h * 0.65)

                z = self.get_depth_median(u, v, radius=5)
                rospy.loginfo_throttle(
                    1.0,
                    "candidate center=(%d,%d), depth=%s",
                    u, v, str(z)
                )

                cv2.circle(img, (u, v), 5, (0, 0, 255), -1)

                if z is None:
                    continue

                X, Y, Z = self.pixel_to_3d(u, v, z)
                pt_map = self.to_map(X, Y, Z)
                if pt_map is None:
                    continue

                map_x = pt_map.point.x
                map_y = pt_map.point.y

                self.update_tracks(digit, map_x, map_y, score)

                # Draw accepted candidate in green
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{digit} ({score:.2f})",
                    (x, max(20, y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"x={map_x:.2f}, y={map_y:.2f}",
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            self.confirm_tracks()

            if (rospy.Time.now() - self.last_print_time).to_sec() > 2.0:
                counts, clustered = self.get_top4_digit_counts()
                rospy.loginfo("Current clustered top4 counts: %s", counts)
                rospy.loginfo(
                    "Clustered boxes total: %d | Raw confirmed total: %d",
                    len(clustered),
                    len(self.confirmed)
                )
                self.last_print_time = rospy.Time.now()

            cv2.imshow(self.window, img)
            cv2.imshow(self.mask_window, mask)
            cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    node = BoxCounter()
    try:
        node.spin()
    finally:
        node.print_final_result()
