#!/usr/bin/env python3
# yolo_detector.py

import time
from typing import Union

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)
from cv_bridge import CvBridge

import numpy as np
import cv2
import torch
from ultralytics import YOLO


def _normalize_device_param(device_param: Union[str, int]) -> str:
    """
    Accepts "cpu", "cuda:0", "auto" or an int (e.g., 0).
    Returns a torch/Ultralytics-friendly string and falls back to CPU if needed.
    """
    if isinstance(device_param, int):
        # Back-compat: user passed 0/1/...
        if torch.cuda.is_available():
            return f"cuda:{device_param}"
        return "cpu"

    # String path
    s = str(device_param).strip().lower()
    if s in ("cpu",):
        return "cpu"
    if s in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    if s.startswith("cuda"):
        # Respect explicit CUDA request if available; else fallback
        if torch.cuda.is_available():
            return s
        return "cpu"
    return "cpu"


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        # ---------------- Parameters ----------------
        self.declare_parameter("model_path", "")
        self.declare_parameter(
            "image_topic", "/realsense/camera/color/image_raw/compressed"
        )
        # Accept string ("cpu", "cuda:0", "auto") or int (0/1/...)
        self.declare_parameter("device", "auto") #auto means decide yourself GPU or CPU. 
        self.declare_parameter("imgsz", 768)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.6)
        self.declare_parameter("publish_image", True)
        self.declare_parameter("overlay_topic", "/yolo/overlay") #is topic me image dikhegi with bbox
        self.declare_parameter("detections_topic", "/yolo/detections") #is topic mein sirf details hai without image.

        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        device_param = self.get_parameter("device").get_parameter_value()
        # Simplified parameter handling
        try:
            device_value = device_param.string_value
            if not device_value:  # Empty string
                device_value = device_param.integer_value if hasattr(device_param, 'integer_value') else "auto"
        except AttributeError:
            try:
                device_value = device_param.integer_value
            except AttributeError:
                device_value = "auto"

        self.device = _normalize_device_param(device_value)
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value) #image size
        self.conf = float(self.get_parameter("conf").get_parameter_value().double_value) #confidence threshold
        self.iou = float(self.get_parameter("iou").get_parameter_value().double_value) #IoU threshold
        self.pub_img = bool(
            self.get_parameter("publish_image").get_parameter_value().bool_value
        )
        overlay_topic = (
            self.get_parameter("overlay_topic").get_parameter_value().string_value
        )
        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )

        # ---------------- Model ----------------
        try:
            self.model = YOLO(model_path)
            # Jb predict chlta hai to ye apne aap model ko device pe le jata hai.
            # But we can also do it manually here if needed, par abhi hm wo nhi kar rhe..
            self.model.fuse()  # small speedup
            # Conv + BN → ek single optimized Conv layer ban jaata hai.
            self.get_logger().info(
                f"YOLO model loaded and fused: {model_path} | device={self.device}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed loading YOLO model: {e}")
            raise

        # ---------------- Bridge & pubs/subs ----------------
        self.bridge = CvBridge() #ROS Image messages ↔ OpenCV images (numpy array).
        self.pub_det = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.pub_overlay = None
        if self.pub_img:
            self.pub_overlay = self.create_publisher(Image, overlay_topic, 10)

        # Camera streams should use sensor-data QoS (best-effort, small queue)
        self.sub = self.create_subscription(
            CompressedImage, image_topic, self.cb_compressed, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Subscribing to compressed images: {image_topic} (sensor-data QoS)"
        )
        self.get_logger().info(
            f"Running | imgsz={self.imgsz} | conf={self.conf} | iou={self.iou}"
        )

        # ---------------- Perf stats ----------------
        self._frame_count = 0
        self._last_fps_log = time.time()

    # ---------------- Callback ----------------
    def cb_compressed(self, msg: CompressedImage):
        # Spammy logs → use debug; switch to rclpy logging config to see
        self.get_logger().debug(
            f"Compressed frame: format={msg.format}, bytes={len(msg.data)}"
        )

        try:
            buf = np.frombuffer(msg.data, np.uint8)
            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # Decodes to BGR, 
            if frame_bgr is None:
                self.get_logger().warn("cv2.imdecode returned None, skipping frame")
                return
            self._run_inference(frame_bgr, msg.header)
        except Exception as e:
            self.get_logger().error(f"Compressed image callback error: {e}")

    # ---------------- Inference & Publish ----------------
    def _run_inference(self, frame_bgr: np.ndarray, header):
        t0 = time.time()
        try:
            results = self.model.predict(
                source=frame_bgr,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"Model predict() failed: {e}")
            return

        if not results:
            self.get_logger().debug("Empty results list, skipping")
            return

        r = results[0]  # Ultralytics returns List[Results]; work on first
        boxes = r.boxes  # Boxes object, bbox + attributes. 
        names = r.names if hasattr(r, "names") else None  # check names attribute is in r object or not. 

        det_arr = Detection2DArray() #information type/container for 2D multi-detections (could be one detection or multiple.)
        det_arr.header = header

        overlay = frame_bgr.copy()

        try:
            if boxes is not None and len(boxes) > 0:
                # Tensors: (N,4), (N,), (N,)
                xyxy = boxes.xyxy  # x1,y1,x2,y2 gives co-ordinates of bbox.
                conf = boxes.conf if hasattr(boxes, "conf") else None
                cls = boxes.cls if hasattr(boxes, "cls") else None

                n = xyxy.shape[0] # .shape(3,4) - means 3 boxes detected and each box has 4 coordinates.
                for i in range(n):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()] #tolist - from tensor to normal list.
                    score = float(conf[i].item()) if conf is not None else 0.0 #.item() - from tensor single to normal single value.
                    cls_id = int(cls[i].item()) if cls is not None else 0
                    label = None
                    if isinstance(names, dict): #YOLOv8 trained on COCO dataset, so names is dict.
                        label = names.get(cls_id, str(cls_id))
                    elif isinstance(names, list) and 0 <= cls_id < len(names): #Custom trained model, so names is list.
                        label = names[cls_id]
                    else:
                        label = str(cls_id)

                    # ---- vision_msgs Detection2D ----
                    det = Detection2D() #for single detection.
                    det.header = header

                    hyp = ObjectHypothesisWithPose()
                    # Put the human-readable label in class_id; downstream can map as needed
                    hyp.hypothesis.class_id = label
                    hyp.hypothesis.score = score
                    det.results.append(hyp)

                    bb = BoundingBox2D()
                    bb.center.position.x = (x1 + x2) / 2.0
                    bb.center.position.y = (y1 + y2) / 2.0
                    bb.size_x = max(0.0, x2 - x1)
                    bb.size_y = max(0.0, y2 - y1)
                    det.bbox = bb

                    det_arr.detections.append(det)

                    # ---- Draw overlay ----
                    cv2.rectangle(
                        overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    cv2.putText(
                        overlay,
                        f"{label} {score:.2f}",
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        except Exception as e:
            self.get_logger().error(f"Postprocessing error: {e}")

        # ---- Publish detections
        try:
            self.pub_det.publish(det_arr)
        except Exception as e:
            self.get_logger().error(f"Publishing detections failed: {e}")

        # ---- Publish overlay (raw Image)
        # pub_img - flag for data publish krna(ROS messages) jisme sirf class_id, confidence, bbox (center+size).
        # pub_overlay - flag for data publish krna(ROS messages) jisme image with bbox draw hua h.
        if self.pub_img and self.pub_overlay is not None:
            try:
                msg_out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8") #OpenCV image (overlay) to ROS Image
                msg_out.header = header
                self.pub_overlay.publish(msg_out)
            except Exception as e:
                self.get_logger().error(f"Publishing overlay failed: {e}")

        # ---- Periodic FPS log
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_log >= 2.0:  # every ~2s
            dt = now - self._last_fps_log
            fps = self._frame_count / max(1e-6, dt)
            self.get_logger().info(f"Inference FPS: {fps:.1f}")
            self._frame_count = 0
            self._last_fps_log = now


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
