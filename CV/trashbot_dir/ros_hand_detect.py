# SOURCE REFERENCES USED IN THIS FILE:
# - Ultralytics YOLO Python API:
#   https://docs.ultralytics.com/usage/python/
# - Ultralytics Results/Boxes:
#   https://docs.ultralytics.com/reference/engine/results/
# - ROS 2 Executors conceptual documentation:
#   https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html
# - ROS sensor_msgs/Image:
#   https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html
# - ROS sensor_msgs/CameraInfo:
#   https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
# - ROS image encoding constants:
#   https://docs.ros.org/en/rolling/p/sensor_msgs/generated/program_listing_file_include_sensor_msgs_image_encodings.hpp.html
# - ROS REP-118 Depth Images:
#   https://www.ros.org/reps/rep-0118.html
# - OpenCV:
#   https://opencv.org
# - NumPy:
#   https://numpy.org

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO


MODEL_PATH = Path("/home/pi/E90_ws/src/CV/trashbot_dir/yolo11.pt")

# For RGB imaging
COLOR_TOPIC = "/camera/camera/color/image_raw"
# For Depth when calculating coordinates from camera
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
# For our camera intrinsics
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

# Params to alter for best performance
YOLO_IMGSZ = 960
YOLO_CONF = 0.10
YOLO_IOU = 0.45

PROCESS_INTERVAL_SECONDS = 0.05

DEPTH_KERNEL = 11
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 20.00


class HandDetectionResult:
    # This class is just a simple container for the final detection result
    # the found varaible tells us whether a hand was detected, 
    # and xyz_m stores the 3D position if we could compute it
    def __init__(self, found=False, xyz_m=None):
        self.found = found
        self.xyz_m = xyz_m


def ros_color_to_bgr(msg):
    # Based on ROS Image message layout docs: width, height, step, and raw data layout
    flat = np.frombuffer(msg.data, dtype=np.uint8)
    frame = flat.reshape(msg.height, msg.step)
    frame = frame[:, : msg.width * 3]
    frame = frame.reshape(msg.height, msg.width, 3)

    # ROS color images come in RGB here, but OpenCV expects BGR for most operations
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def ros_depth_to_array(msg):
    # Based on ROS Image message layout docs for raw image buffers
    flat = np.frombuffer(msg.data, dtype=np.uint16)
    frame = flat.reshape(msg.height, msg.step // 2)

    # Depth is stored as 16-bit values, so we reshape a little differently than the color image
    return frame[:, : msg.width].copy()


def clamp(value, lower, upper):
    # Forces the box's coordinates to fall within the image dimensions
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def median_depth_meters(depth_frame, center_uv):
    # Uses a small local depth window around the detection center and takes the median
    # to reduce noisy depth readings
    u = center_uv[0]
    v = center_uv[1]
    half = DEPTH_KERNEL // 2

    height = depth_frame.shape[0]
    width = depth_frame.shape[1]

    # Build a small square region around the center of the detected hand
    x1 = max(0, u - half)
    y1 = max(0, v - half)
    x2 = min(width, u + half + 1)
    y2 = min(height, v + half + 1)

    roi = depth_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # RealSense depth is typically in millimeters, so convert to meters here
    roi_m = roi.astype(np.float32) / 1000.0
    # Below ensures that the depths are finite and within our expected/allowable range
    valid = roi_m[np.isfinite(roi_m)]
    valid = valid[(valid >= MIN_DEPTH_M) & (valid <= MAX_DEPTH_M)]

    # If all values are bad or out of range, we give up on the depth estimate
    if valid.size == 0:
        return None

    return float(np.median(valid))


def deproject_to_xyz(u, v, depth_m, camera_info):
    # Standard pinhole-camera back-projection using CameraInfo intrinsics
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])

    # Learned in Swarthmore's computer vision course
    # This takes a pixel location plus depth and turns it into a 3D point in camera coordinates
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    return (x, y, z)


class HandDetectNode(Node):
    def __init__(self):
        # ROS 2 Python node/subscription pattern follows rclpy tutorials/API usage
        super().__init__("ros_hand_detect")

        # Checks for valid model path
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        # YOLO model loading/prediction flow follows Ultralytics Python usage docs
        self.model = YOLO(MODEL_PATH)
        self.lock = threading.Lock()

        # These hold the newest data from each topic.
        # We keep updating them as messages come in so detection can always use the latest inputs
        self.color_frame = None
        self.depth_frame = None
        self.camera_info = None

        # ROS subscriptions based on standard rclpy create_subscription usage
        # One topic for color, one for depth, and one for the camera intrinsics
        self.create_subscription(Image, COLOR_TOPIC, self.on_color, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.on_camera_info, 10)

    # Preprocessing the Frames for proper channels and updates shared variable
    def on_color(self, msg):
        frame = ros_color_to_bgr(msg)
        with self.lock:
            self.color_frame = frame

    # Uses our Depth preprocessing and updates shared variable
    def on_depth(self, msg):
        frame = ros_depth_to_array(msg)
        with self.lock:
            self.depth_frame = frame

    def on_camera_info(self, msg):
        with self.lock:
            self.camera_info = msg

    def get_latest_inputs(self):
        with self.lock:
            color = self.color_frame
            depth = self.depth_frame
            camera_info = self.camera_info

            # Return copies to ensure there is no contamination from
            # new variable updates
            if color is not None:
                color = color.copy()

            if depth is not None:
                depth = depth.copy()

        return color, depth, camera_info

    def detect_hand(self):
        # Pull the newest color frame, depth frame, as well as the camera intrinsics
        color, depth, camera_info = self.get_latest_inputs()

        if color is None:
            return HandDetectionResult()

        # YOLO inference arguments and result access follow Ultralytics predict/results docs
        # We run detection on just the current color frame
        results = self.model.predict(
            source=color,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None:
            return HandDetectionResult()

        if len(result.boxes) == 0:
            return HandDetectionResult()

        # Extracting xyxy bounding boxes and confidences follows Ultralytics result tensor usage
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()

        # If multiple hands were detected, pick the one YOLO had the highest confidence in
        best_index = int(confidences.argmax())
        box = boxes[best_index]

        height = color.shape[0]
        width = color.shape[1]

        # Clamp box corners so they stay inside the image bounds
        x1 = clamp(int(box[0]), 0, width - 1)
        y1 = clamp(int(box[1]), 0, height - 1)
        x2 = clamp(int(box[2]), 0, width - 1)
        y2 = clamp(int(box[3]), 0, height - 1)

        # Use the center of the bounding box as the point where we sample depth
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2

        # If we only have color, we can still say a hand was found,
        # but we cannot return a 3D position yet
        # Do we have a depth frame?
        if depth is None:
            return HandDetectionResult(found=True, xyz_m=None)

        depth_m = median_depth_meters(depth, (center_u, center_v))

        # Could we successfuly use that depth frame?
        if depth_m is None:
            return HandDetectionResult(found=True, xyz_m=None)

        # Camera intrinsics are required to convert from pixel coordinates into real-world xyz
        if camera_info is None:
            return HandDetectionResult(found=True, xyz_m=None)

        xyz_m = deproject_to_xyz(center_u, center_v, depth_m, camera_info)
        return HandDetectionResult(found=True, xyz_m=xyz_m)


class HandDetectRunner:
    # This class handles the higher-level execution flow:
    # start ROS, spin the node in the background, and keep checking until a hand is found
    def __init__(self):
        rclpy.init()
        self.node = HandDetectNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # Separate executor thread is a standard ROS 2 Python pattern for spinning a node
        # This lets ROS callbacks keep running while our main logic does its own loop
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def shutdown(self):
        # Shutdown function so the background ROS thread does not get left running
        self.executor.shutdown()
        self.thread.join(timeout=2.0)
        self.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    def wait_for_first_frame(self):
        # Wait until at least one color frame has been recieved before trying to detect anything
        while True:
            color, _, _ = self.node.get_latest_inputs()
            if color is not None:
                return
            time.sleep(0.05)

    def run_until_detected(self):
        self.wait_for_first_frame()

        # Keep checking frames until YOLO says it found a hand
        while True:
            result = self.node.detect_hand()
            if result.found:
                return result
            time.sleep(PROCESS_INTERVAL_SECONDS)


def detect_hand():
    # Wrapper so the runner always gets cleaned up,
    # even if something goes wrong during detection or code was cntrl + c
    runner = HandDetectRunner()
    try:
        return runner.run_until_detected()
    finally:
        runner.shutdown()


def main():
    result = detect_hand()
    print(json.dumps(result.__dict__, indent=2))

    # Return code 0 means success, 1 means no hand was found
    if result.found:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())