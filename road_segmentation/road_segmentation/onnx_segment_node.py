#!/usr/bin/env python3
# Author: Sahil Rajesh Patil

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import onnxruntime as ort
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import os
import road_segmentation

from rclpy.qos import qos_profile_sensor_data


class ONNXRoadSeg(Node):
    def __init__(self):
        super().__init__("onnx_road_seg")

        self.bridge = CvBridge()

        # -----------------------------
        # Load ONNX model
        # -----------------------------
        pkg_dir = os.path.dirname(road_segmentation.__file__)
        model_path = os.path.join(pkg_dir, "deeplabv3plus_road.onnx")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # -----------------------------
        # ROS interfaces (RGB SAME)
        # -----------------------------
        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_cb,
            qos_profile_sensor_data
        )

        # -----------------------------
        # Depth subscriber (RAW depth that you said works)
        # -----------------------------
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self.depth_cb,
            qos_profile_sensor_data
        )

        self.pub_mask = self.create_publisher(Image, "/road_seg/mask", 10)
        self.pub_overlay = self.create_publisher(Image, "/road_seg/overlay", 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/road_seg/lane_geometry", 10
        )

        # NEW: depth + mask visualization topic
        self.pub_depth_masked = self.create_publisher(Image, "/road_seg/depth_masked", 10)

        # store latest depth
        self.last_depth_msg = None

        # for width in meters: width_m ≈ (px * z) / fx
        self.fx = 607.3188

        self._last_log_ns = 0

        self.get_logger().info(
            "✅ ONNX Road Segmentation + Lane Geometry Node Started"
        )

    def depth_cb(self, msg: Image):
        self.last_depth_msg = msg

    # ----------------------------------------------------
    # CALLBACK (RGB unchanged + extra depth processing)
    # ----------------------------------------------------
    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame.shape

        # -----------------------------
        # Preprocess
        # -----------------------------
        img = cv2.resize(frame, (512, 512))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # ONNX inference
        # -----------------------------
        out = self.session.run(None, {self.input_name: img})[0]
        mask = np.argmax(out, axis=1)[0].astype(np.uint8)

        lane_mask = (mask == 1).astype(np.uint8) * 255  # 512x512

        # -----------------------------
        # Publish mask (same)
        # -----------------------------
        color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        color_mask[lane_mask == 255] = (0, 255, 0)

        mask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding="bgr8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # -----------------------------
        # Overlay (same)
        # -----------------------------
        color_mask_rs = cv2.resize(color_mask, (w, h))
        overlay = cv2.addWeighted(frame, 0.7, color_mask_rs, 0.3, 0)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        # -----------------------------
        # Geometry (same)
        # -----------------------------
        self.extract_and_publish_geometry(lane_mask, msg.header)

        # -----------------------------
        # NEW: Copy same mask onto DEPTH and compute distance/width
        # -----------------------------
        if self.last_depth_msg is not None:
            self.apply_mask_on_depth_and_publish(lane_mask, self.last_depth_msg)

    # ----------------------------------------------------
    # DEPTH + MASK
    # ----------------------------------------------------
    def apply_mask_on_depth_and_publish(self, lane_mask_512, depth_msg: Image):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        dh, dw = depth.shape[:2]

        # resize mask to depth size
        lane_mask_depth = cv2.resize(lane_mask_512, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # convert depth to meters (RealSense raw depth usually 16UC1 in mm)
        if depth_msg.encoding == "16UC1":
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0

        # create colormap visualization only in masked region
        max_range = 10.0
        d = np.clip(depth_m, 0.0, max_range)
        d8 = (255.0 * (d / max_range)).astype(np.uint8)
        cm = cv2.applyColorMap(d8, cv2.COLORMAP_JET)

        out = np.zeros_like(cm)
        out[lane_mask_depth == 255] = cm[lane_mask_depth == 255]

        depth_vis_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        depth_vis_msg.header = depth_msg.header
        self.pub_depth_masked.publish(depth_vis_msg)

        # ---------- metrics ----------
        ys, xs = np.where(lane_mask_depth == 255)
        if xs.size < 200:
            return

        vals = depth_m[ys, xs]
        vals = vals[vals > 0.1]
        if vals.size < 200:
            return

        # distance from camera: near (bottom) and far (top)
        y_thr = int(0.85 * lane_mask_depth.shape[0])
        near_sel = ys >= y_thr
        near_vals = depth_m[ys[near_sel], xs[near_sel]]
        near_vals = near_vals[near_vals > 0.1]
        near_m = float(np.median(near_vals)) if near_vals.size > 0 else float(np.median(vals))

        y_min = int(np.min(ys))
        far_xs = xs[ys == y_min]
        far_vals = depth_m[y_min, far_xs] if far_xs.size > 0 else vals
        far_vals = far_vals[far_vals > 0.1]
        far_m = float(np.median(far_vals)) if far_vals.size > 0 else float(np.median(vals))

        # width at mid row
        mid_y = int(0.60 * lane_mask_depth.shape[0])
        row = lane_mask_depth[mid_y, :]
        lane_xs = np.where(row == 255)[0]
        if lane_xs.size >= 2:
            width_px = int(lane_xs.max() - lane_xs.min())
            row_depth = depth_m[mid_y, lane_xs]
            row_depth = row_depth[row_depth > 0.1]
            z = float(np.median(row_depth)) if row_depth.size > 0 else near_m
            width_m = (width_px * z) / float(self.fx)
        else:
            width_px = 0
            width_m = 0.0

        # -----------------------------
        # ✅ ADDED: width only in bottom 10% of image (simple)
        # -----------------------------
        h = lane_mask_depth.shape[0]
        y0 = int(0.90 * h)
        roi_mask = lane_mask_depth[y0:h, :]

        ys_roi, xs_roi = np.where(roi_mask == 255)
        if xs_roi.size >= 50:
            min_x = int(xs_roi.min())
            max_x = int(xs_roi.max())
            width_bottom_px = max_x - min_x

            ys_full = ys_roi + y0
            z_vals = depth_m[ys_full, xs_roi]
            z_vals = z_vals[z_vals > 0.1]

            if z_vals.size > 0:
                z_bottom = float(np.median(z_vals))
                width_bottom_m = (float(width_bottom_px) * z_bottom) / float(self.fx)
            else:
                z_bottom = 0.0
                width_bottom_m = 0.0
        else:
            width_bottom_px = 0
            z_bottom = 0.0
            width_bottom_m = 0.0

        # log once per second
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_log_ns) > 1_000_000_000:
            self._last_log_ns = now_ns
            self.get_logger().info(
                f"[DEPTH+MASK] near={near_m:.2f}m far={far_m:.2f}m "
                f"width@mid={width_m:.2f}m (px={width_px}) | "
                f"width@bottom10%={width_bottom_m:.2f}m (px={width_bottom_px}, z={z_bottom:.2f}m)"
            )

    # ----------------------------------------------------
    # GEOMETRY EXTRACTION (unchanged)
    # ----------------------------------------------------
    def extract_and_publish_geometry(self, lane_mask, header):
        edges = cv2.Canny(lane_mask, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) < 2:
            return

        contours = sorted(contours, key=lambda c: np.mean(c[:, 0, 0]))

        left = contours[0]
        right = contours[-1]

        min_len = min(len(left), len(right))
        center_pts = []

        for i in range(min_len):
            lx, ly = left[i][0]
            rx, ry = right[i][0]
            cx = (lx + rx) / 2.0
            cy = (ly + ry) / 2.0
            center_pts.append((cx, cy))

        marker_array = MarkerArray()

        marker_array.markers.append(
            self.make_marker_from_contour(left, header, 0, 1.0, 0.0, 0.0)
        )
        marker_array.markers.append(
            self.make_marker_from_contour(right, header, 1, 0.0, 0.0, 1.0)
        )
        marker_array.markers.append(
            self.make_marker_from_points(center_pts, header, 2, 0.0, 1.0, 0.0)
        )

        self.pub_markers.publish(marker_array)

    # ----------------------------------------------------
    # MARKER HELPERS (unchanged)
    # ----------------------------------------------------
    def make_marker_from_contour(self, contour, header, mid, r, g, b):
        pts = [(p[0][0], p[0][1]) for p in contour]
        return self.make_marker_from_points(pts, header, mid, r, g, b)

    def make_marker_from_points(self, pts, header, mid, r, g, b):
        marker = Marker()
        marker.header = header
        marker.ns = "lane_geometry"
        marker.id = mid
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        for (x, y) in pts:
            pt = Point()
            pt.x = x / 100.0
            pt.y = y / 100.0
            pt.z = 0.0
            marker.points.append(pt)

        return marker


def main():
    rclpy.init()
    node = ONNXRoadSeg()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()