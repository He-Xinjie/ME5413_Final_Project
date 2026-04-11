#!/usr/bin/env python3
"""
3D Box Detector - Detects numbered boxes using YOLO11 + RGB-D camera.

Pipeline:
  1. YOLO inference on RGB to detect digit + bounding box
  2. Filter by bounding box size and depth validity
  3. Look up depth at detection center -> back-project to 3D (camera frame)
  4. TF transform to map frame -> global (x,y,z) of the box
  5. Filter by map-frame height (z) to reject floor/sky outliers
  6. Distance-based clustering for deduplication (same physical box = 1 candidate)
     - Same digit: large merge radius (handles viewing-angle offset)
     - Different digit: small merge radius (prevents merging neighbors)
  7. Periodic candidate merge pass to clean up duplicates
  8. Each candidate accumulates votes per digit across frames

Subscribes:
  /front/image_raw       (RGB)
  /front/depth/image_raw (Depth, 32FC1 or 16UC1)
  /front/camera_info     (CameraInfo for intrinsics)

Publishes:
  /box_detector/image      (visualization)
  /box_detector/counts     (deduplicated counts string)
  /mission_planner/min_box_id (digit with minimum count)
"""

import rospy
import cv2
import numpy as np
import os
import math
import time
import tf2_ros
import tf2_geometry_msgs
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, ColorRGBA, Empty
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from ultralytics import YOLO


class BoxCandidate:
    """A physical box in the global map with vote-based digit identification."""
    def __init__(self, map_x, map_y, map_z, digit):
        self.x = map_x
        self.y = map_y
        self.z = map_z
        self.obs_count = 1
        self.votes = {}
        self.last_seen = time.time()
        self.add_vote(digit)

    def add_vote(self, digit, new_x=None, new_y=None):
        self.votes[digit] = self.votes.get(digit, 0) + 1
        self.last_seen = time.time()
        # Only update position if new observation is reasonably close
        # Prevents a single bad depth reading from dragging the position away
        if new_x is not None and new_y is not None:
            dist = math.sqrt((self.x - new_x)**2 + (self.y - new_y)**2)
            if dist < 2.0:
                self.obs_count += 1
                alpha = 1.0 / self.obs_count
                self.x = self.x + alpha * (new_x - self.x)
                self.y = self.y + alpha * (new_y - self.y)

    def best_digit(self):
        return max(self.votes, key=self.votes.get)

    def best_digit_ratio(self):
        """Ratio of best digit votes to total votes."""
        total = self.total_votes()
        if total == 0:
            return 0.0
        return self.votes[self.best_digit()] / total

    def total_votes(self):
        return sum(self.votes.values())

    def is_confirmed(self, min_votes=3):
        return self.total_votes() >= min_votes and self.best_digit_ratio() >= 0.5

    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)

    def merge_from(self, other):
        """Absorb another candidate into this one."""
        for digit, count in other.votes.items():
            self.votes[digit] = self.votes.get(digit, 0) + count
        # Weighted average position by observation count
        total = self.obs_count + other.obs_count
        self.x = (self.x * self.obs_count + other.x * other.obs_count) / total
        self.y = (self.y * self.obs_count + other.y * other.obs_count) / total
        self.obs_count = total
        self.last_seen = max(self.last_seen, other.last_seen)


class BoxDetector3D:
    def __init__(self):
        rospy.init_node('box_detector_3d', anonymous=False)

        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera intrinsics (filled by camera_info callback)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # ---- Clustering parameters ----
        # Two-tier radius: small for different digits, large for same digit.
        # Same box seen from opposite directions can shift 2-4m due to:
        #   depth hitting different faces + AMCL drift + viewing angle
        # But different boxes of the same digit are always >5m apart in this scene.
        self.cluster_radius = 2.5           # merge radius for DIFFERENT digits
        self.same_digit_radius = 5.0        # merge radius for SAME digit
        self.merge_check_interval = 5.0     # seconds between merge passes
        self.last_merge_time = 0.0
        self.min_confirm_votes = 3
        self.candidates = []

        # ---- Detection parameters ----
        self.yolo_conf = rospy.get_param('~yolo_conf', 0.6)
        self.yolo_iou = rospy.get_param('~yolo_iou', 0.45)
        self.min_depth = 0.5
        self.max_depth = 7.0
        self.depth_patch_size = 7
        self.min_bbox_area = 1500
        self.min_bbox_side = 30

        # ---- Height (z) filter in map frame ----
        self.min_map_z = -0.5
        self.max_map_z = 4.0

        # ---- Cooldown: time-based, per-candidate ----
        self.cooldown_seconds = 1.0

        # Load YOLO model
        model_path = rospy.get_param('~model_path', '')
        if not model_path:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'Final.pt'
            )
        rospy.loginfo("[BoxDet3D] Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)
        rospy.loginfo("[BoxDet3D] YOLO model loaded. Classes: %s", self.model.names)

        # Publishers
        self.pub_min_box = rospy.Publisher(
            '/mission_planner/min_box_id', String, queue_size=1, latch=True)
        self.pub_viz = rospy.Publisher(
            '/box_detector/image', Image, queue_size=1)
        self.pub_counts = rospy.Publisher(
            '/box_detector/counts', String, queue_size=1)
        self.pub_markers = rospy.Publisher(
            '/box_detector/markers', MarkerArray, queue_size=1)

        # Detection gate: stairway off, lower/upper on. On "upper" we also
        # clear candidates so each room scan starts from a clean slate.
        self.detection_enabled = True
        self.sub_zone_cmd = rospy.Subscriber(
            '/nav_zone/command', String, self.zone_cmd_callback)

        # External reset: nav_test publishes on this before each upper room
        # scan so only that room's box gets voted on.
        self.sub_reset = rospy.Subscriber(
            '/box_detector/reset', Empty, self.reset_callback)

        # Camera info subscriber
        self.sub_cam_info = rospy.Subscriber(
            '/front/camera_info', CameraInfo, self.camera_info_callback)

        # Synchronized RGB + Depth subscribers
        rgb_sub = message_filters.Subscriber('/front/image_raw', Image)
        depth_sub = message_filters.Subscriber('/front/depth/image_raw', Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.1)
        sync.registerCallback(self.rgbd_callback)

        rospy.loginfo("[BoxDet3D] Initialized. Waiting for RGB-D data...")

    def zone_cmd_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == "lower":
            if not self.detection_enabled:
                rospy.loginfo("[BoxDet3D] Detection ENABLED (lower phase)")
            self.detection_enabled = True
        elif cmd == "upper":
            self.candidates = []
            self.detection_enabled = True
            rospy.loginfo("[BoxDet3D] Detection ENABLED + candidates cleared (upper phase)")
        elif cmd in ("stairway", "stop"):
            if self.detection_enabled:
                rospy.loginfo("[BoxDet3D] Detection DISABLED (phase=%s)", cmd)
            self.detection_enabled = False

    def reset_callback(self, msg):
        rospy.loginfo("[BoxDet3D] Candidates cleared by external reset (%d removed)",
                      len(self.candidates))
        self.candidates = []

    def camera_info_callback(self, msg):
        if self.fx is not None:
            return
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        rospy.loginfo("[BoxDet3D] Camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                      self.fx, self.fy, self.cx, self.cy)

    def pixel_to_3d(self, u, v, depth):
        if self.fx is None:
            return None
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return (x, y, z)

    def transform_to_map(self, x, y, z, stamp=None):
        # Use the image capture timestamp so fast rotation doesn't misplace
        # the 3D point: the pose at image time is what the pixel actually saw.
        pt = PointStamped()
        pt.header.frame_id = "front_camera_optical"
        pt.header.stamp = stamp if stamp is not None else rospy.Time(0)
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z
        try:
            pt_map = self.tf_buffer.transform(pt, "map", rospy.Duration(0.3))
            return pt_map.point.x, pt_map.point.y, pt_map.point.z
        except Exception:
            # Stamped lookup failed (e.g. TF buffer gap); fall back to latest
            try:
                pt.header.stamp = rospy.Time(0)
                pt_map = self.tf_buffer.transform(pt, "map", rospy.Duration(0.2))
                return pt_map.point.x, pt_map.point.y, pt_map.point.z
            except Exception as e:
                rospy.logwarn_throttle(5.0, "[BoxDet3D] TF to map failed: %s", str(e))
                return None

    def get_depth_at(self, depth_image, u, v):
        """Get median depth in a patch around (u,v), filtering NaN/inf."""
        h, w = depth_image.shape[:2]
        half = self.depth_patch_size // 2
        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)
        patch = depth_image[v0:v1, u0:u1].flatten()
        valid = patch[np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def find_or_create_candidate(self, map_x, map_y, map_z, digit):
        """Find existing candidate of the SAME digit within same_digit_radius.
        Different-digit candidates are never merged, so two boxes with
        different numbers spawned close together stay as separate counts.
        Returns (candidate, is_new, is_on_cooldown)."""
        now = time.time()
        best_cand = None
        best_dist = float('inf')

        for cand in self.candidates:
            # Only merge into candidates that already agree on this digit.
            if cand.best_digit() != digit:
                continue
            d = cand.distance_to(map_x, map_y)
            if d < self.same_digit_radius and d < best_dist:
                best_dist = d
                best_cand = cand

        if best_cand is not None:
            # Check per-candidate cooldown
            elapsed = now - best_cand.last_seen
            if elapsed < self.cooldown_seconds:
                return best_cand, False, True
            best_cand.add_vote(digit, new_x=map_x, new_y=map_y)
            return best_cand, False, False

        # New candidate
        cand = BoxCandidate(map_x, map_y, map_z, digit)
        self.candidates.append(cand)
        return cand, True, False

    def merge_nearby_candidates(self):
        """Periodic pass: merge any two candidates whose best_digit matches
        and are within same_digit_radius. Handles cases where two candidates
        were created before either had enough votes to establish a digit."""
        merged = True
        while merged:
            merged = False
            for i in range(len(self.candidates)):
                if merged:
                    break
                for j in range(i + 1, len(self.candidates)):
                    ci = self.candidates[i]
                    cj = self.candidates[j]
                    d = ci.distance_to(cj.x, cj.y)
                    # Same digit within large radius -> merge
                    if ci.best_digit() == cj.best_digit() and d < self.same_digit_radius:
                        # Keep the one with more votes
                        if ci.total_votes() >= cj.total_votes():
                            ci.merge_from(cj)
                            self.candidates.pop(j)
                        else:
                            cj.merge_from(ci)
                            self.candidates.pop(i)
                        rospy.loginfo(
                            "[BoxDet3D] MERGED two #%d candidates (dist=%.1fm), %d candidates remain",
                            ci.best_digit(), d, len(self.candidates))
                        merged = True
                        break

    def get_deduplicated_counts(self):
        """Get box counts from confirmed candidates only."""
        counts = {}
        for cand in self.candidates:
            if cand.is_confirmed(self.min_confirm_votes):
                digit = cand.best_digit()
                counts[digit] = counts.get(digit, 0) + 1
        return counts

    def rgbd_callback(self, rgb_msg, depth_msg):
        if self.fx is None:
            return

        # When not in lower-level phase, skip YOLO and voting but still
        # publish the raw camera frame and existing confirmed markers so
        # the image topic stays alive and lower-phase results stay visible.
        if not self.detection_enabled:
            try:
                raw = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
                cv2.putText(raw, "DETECTION PAUSED", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                self.pub_viz.publish(self.bridge.cv2_to_imgmsg(raw, 'bgr8'))
            except Exception:
                pass
            self.publish_markers()
            return

        # Convert images
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as e:
            rospy.logwarn("[BoxDet3D] RGB convert error: %s", str(e))
            return

        try:
            if depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            rospy.logwarn("[BoxDet3D] Depth convert error: %s", str(e))
            return

        # --- Periodic candidate merge pass ---
        now = time.time()
        if now - self.last_merge_time > self.merge_check_interval:
            self.merge_nearby_candidates()
            self.last_merge_time = now

        # --- YOLO inference ---
        results = self.model.predict(
            cv_image,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=640,
            verbose=False
        )

        viz_image = cv_image.copy()
        stamp = rgb_msg.header.stamp

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # Extract detection info
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                digit = int(self.model.names[cls_id])

                # ---- Filter 1: bounding box size ----
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_area = bbox_w * bbox_h
                if bbox_area < self.min_bbox_area or bbox_w < self.min_bbox_side or bbox_h < self.min_bbox_side:
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(viz_image, "#%d too small" % digit,
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                    continue

                # Bounding box center for depth lookup
                center_u = (x1 + x2) // 2
                center_v = (y1 + y2) // 2

                color = (0, 255, 0)
                depth = self.get_depth_at(depth_image, center_u, center_v)

                # ---- Filter 2: valid depth ----
                if depth is None:
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(viz_image, "#%d no depth" % digit,
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                    continue

                # ---- 3D projection ----
                pt_3d = self.pixel_to_3d(center_u, center_v, depth)
                if pt_3d is None:
                    continue

                map_result = self.transform_to_map(*pt_3d, stamp)
                if map_result is None:
                    rospy.logwarn_throttle(5.0, "[BoxDet3D] TF to map failed for #%d", digit)
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    continue

                mx, my, mz = map_result

                # ---- Filter 3: map-frame height check ----
                if mz < self.min_map_z or mz > self.max_map_z:
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(viz_image, "#%d bad z=%.1f" % (digit, mz),
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                    continue

                # ---- Clustering & voting ----
                cand, is_new, is_on_cooldown = self.find_or_create_candidate(mx, my, mz, digit)

                if is_on_cooldown:
                    color = (200, 200, 0)
                elif is_new:
                    color = (0, 0, 255)
                    rospy.loginfo(
                        "[BoxDet3D] NEW candidate #%d at map(%.1f, %.1f, %.1f) depth=%.1fm conf=%.2f",
                        digit, mx, my, mz, depth, conf)
                else:
                    color = (255, 165, 0)
                    rospy.loginfo_throttle(5.0,
                        "[BoxDet3D] Vote #%d at map(%.1f, %.1f) -> best=#%d votes=%d ratio=%.0f%%",
                        digit, cand.x, cand.y, cand.best_digit(), cand.total_votes(),
                        cand.best_digit_ratio() * 100)

                # Draw bounding box only for candidates confirmed by >=3 votes
                if cand.is_confirmed(self.min_confirm_votes):
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(viz_image, "#%d %.2f d=%.1fm" % (digit, conf, depth),
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw deduplicated counts overlay
        counts = self.get_deduplicated_counts()
        y_off = 25
        cv2.putText(viz_image, "YOLO 3D Counts:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_off += 25
        for digit in sorted(counts.keys()):
            cv2.putText(viz_image, "  #%d: %d" % (digit, counts[digit]),
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_off += 22

        y_off += 10
        cv2.putText(viz_image, "Candidates: %d (confirmed: %d)" % (
            len(self.candidates),
            sum(1 for c in self.candidates if c.is_confirmed(self.min_confirm_votes))),
            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Publish
        try:
            self.pub_viz.publish(self.bridge.cv2_to_imgmsg(viz_image, 'bgr8'))
        except Exception:
            pass

        self.pub_counts.publish(String(data=str(counts)))

        if counts:
            min_box = min(counts, key=counts.get)
            self.pub_min_box.publish(String(data=str(min_box)))

        self.publish_markers()

    def publish_markers(self):
        marker_array = MarkerArray()

        digit_colors = {
            1: (1.0, 0.0, 0.0),
            2: (0.0, 1.0, 0.0),
            3: (0.0, 0.0, 1.0),
            4: (1.0, 1.0, 0.0),
            5: (1.0, 0.0, 1.0),
            6: (0.0, 1.0, 1.0),
            7: (1.0, 0.5, 0.0),
            8: (0.5, 0.0, 1.0),
            9: (0.0, 0.5, 0.0),
        }

        for i, cand in enumerate(self.candidates):
            if not cand.is_confirmed(self.min_confirm_votes):
                continue
            digit = cand.best_digit()
            r, g, b = digit_colors.get(digit, (1.0, 1.0, 1.0))

            cube = Marker()
            cube.header.frame_id = "map"
            cube.header.stamp = rospy.Time.now()
            cube.ns = "box_cubes"
            cube.id = i
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = cand.x
            cube.pose.position.y = cand.y
            cube.pose.position.z = 0.4
            cube.pose.orientation.w = 1.0
            cube.scale.x = 0.6
            cube.scale.y = 0.6
            cube.scale.z = 0.6
            cube.color = ColorRGBA(r, g, b, 0.7)
            cube.lifetime = rospy.Duration(2.0)
            marker_array.markers.append(cube)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = rospy.Time.now()
            text.ns = "box_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cand.x
            text.pose.position.y = cand.y
            text.pose.position.z = 1.2
            text.pose.orientation.w = 1.0
            text.scale.z = 0.5
            text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text.text = "#%d (%d)" % (digit, cand.total_votes())
            text.lifetime = rospy.Duration(2.0)
            marker_array.markers.append(text)

        self.pub_markers.publish(marker_array)


if __name__ == '__main__':
    try:
        detector = BoxDetector3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
