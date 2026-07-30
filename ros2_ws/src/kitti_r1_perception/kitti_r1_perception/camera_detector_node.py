import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory

class CameraDetectorNode(Node):

    def __init__(self):
        super().__init__('camera_detector_node')
        
        # ⚡ FORCE SIMULATION TIME
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        ])
        
        sys.stdout.reconfigure(line_buffering=True)
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('model_version', 'yolov8n') 
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        
        # Single-shot warmup configuration timer
        self.init_timer = self.create_timer(0.1, self.async_model_init)
    
    def async_model_init(self):
        self.init_timer.cancel()
        
        # Resolve package directories safely
        self.package_share_dir = get_package_share_directory('kitti_r1_perception')
        self.torch_home_dir = os.path.join(self.package_share_dir, 'torch_home')
        os.environ['TORCH_HOME'] = self.torch_home_dir
        
        model_ver = self.get_parameter('model_version').value
        model_path = os.path.join(self.torch_home_dir, f"{model_ver}.onnx")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"ONNX Model missing! Expected at: {model_path}. Please export it first.")
            return

        # Add these environment variables BEFORE creating your InferenceSession
        os.environ["OMP_NUM_THREADS"] = "2"       # Cap CPU helper threads
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE" # Forces threads to yield CPU when waiting

        # ⚡ INITIALIZE HIGH-SPEED ONNX CUDA ENGINE
        opts = ort.SessionOptions()
        
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0
            }),
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f"ONNX Runtime successfully bound execution targets: {self.session.get_providers()}")
        
        # Hardcoded class color tracking arrays for instant layout lookups
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

        # Build Standard Real-time QoS Constraints
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Define ROS infrastructure endpoints
        self.subscription = self.create_subscription(
            Image, '/kitti/camera_left/image_raw', self.listener_callback, realtime_qos, callback_group=self.callback_group
        )
        self.detection_pub = self.create_publisher(Detection2DArray, '/detection/camera_2d', realtime_qos)
        self.annotated_pub = self.create_publisher(Image, '/detection/camera_annotated', realtime_qos)
        self.get_logger().info('ONNX Multi-Shape Perception Pipeline Activated Successfully.')

    def multiclass_nms(self, boxes, scores):
        """Ultra-fast NMS utilizing direct tuple/slicing passes rather than heavy .tolist() conversions."""
        # OpenCV NMS accepts native numpy floats much faster if wrapped precisely
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            scores, 
            float(self.conf_threshold), 
            float(self.iou_threshold)
        )
        if len(indices) > 0:
            return indices.flatten()
        return []

    def listener_callback(self, msg):
        node_start_time = time.perf_counter()
        
        # Calculate Simulation Clock Pre-Callback Age
        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        data_age_ms = (now - msg_time).nanoseconds / 1e6
        
        # Parse image from message payload
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ⚡ TARGET SIZE ALIGNMENT: Match your exact exported baseline shape
        # This takes <0.5ms and prevents any potential dynamic axis rounding errors
        cv_image_resized = cv2.resize(cv_image, (1248, 384), interpolation=cv2.INTER_LINEAR)

        h_orig, w_orig, _ = cv_image_resized.shape

        # ⚡ ONNX INPUT PREPARATION: Zero-copy operations 
        # Convert image matrix layout tracking configurations to match ONNX constraints
        img_rgb = cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose((2, 0, 1)) # HWC to CHW
        img_norm = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
        img_tensor = np.ascontiguousarray(np.expand_dims(img_norm, axis=0), dtype=np.float32)

        # Execute ONNX forward execution map
        inference_start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: img_tensor})
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        # Decode YOLO raw structural matrices [1, 84, Anchors]
        predictions = np.squeeze(outputs[0]).T # Reshapes matrix to [Anchors, 84]
        
        # Filter baseline row indices passing minimum score threshold targets
        scores_matrix = predictions[:, 4:]
        max_scores = np.max(scores_matrix, axis=1)
        valid_mask = max_scores > self.conf_threshold
        
        filtered_preds = predictions[valid_mask]
        
        detections = Detection2DArray()
        detections.header = msg.header
        annotated_frame = cv_image.copy()

        if len(filtered_preds) > 0:
            # Step 1: Decode predictions box formats
            # Raw boxes output format is [center_x, center_y, width, height] relative to the input image size
            cx = filtered_preds[:, 0]
            cy = filtered_preds[:, 1]
            w  = filtered_preds[:, 2]
            h  = filtered_preds[:, 3]
            
            # Recompute target tracking points to fulfill OpenCV top-left bounding box formats
            x1 = cx - (w / 2)
            y1 = cy - (h / 2)
            
            box_array = np.stack([x1, y1, w, h], axis=1)
            score_array = np.max(filtered_preds[:, 4:], axis=1)
            class_array = np.argmax(filtered_preds[:, 4:], axis=1)
            
            # Step 2: Filter overlapped predictions via Vectorized NMS Engine
            keep_indices = self.multiclass_nms(box_array, score_array)
            
            # Step 3: Populate arrays and draw boxes simultaneously
            for idx in keep_indices:
                box = box_array[idx]
                score = float(score_array[idx])
                cls_id = int(class_array[idx])
                
                # Extract absolute bounding box dimensions
                bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                # 1. Populate standard ROS target array matrices
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = bx + (bw / 2.0)
                detection.bbox.center.position.y = by + (bh / 2.0)
                detection.bbox.size_x = bw
                detection.bbox.size_y = bh
                
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(cls_id)
                hypothesis.hypothesis.score = score
                detection.results.append(hypothesis)
                detections.detections.append(detection)
                
                # 2. ⚡ LIGHT-SPEED VISUALIZATION: Vector draw directly onto baseline array
                color = [int(c) for c in self.colors[cls_id % 80]]
                p1 = (int(bx), int(by))
                p2 = (int(bx + bw), int(by + bh))
                
                # Draw outer bounding boxes
                cv2.rectangle(annotated_frame, p1, p2, color, thickness=2)
                
                # Overlay tiny target label tags
                label_text = f"ID:{cls_id} {score:.2f}"
                cv2.putText(annotated_frame, label_text, (p1[0], max(p1[1] - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

        # Publish Output coordinate indices array
        self.detection_pub.publish(detections)
        
        # Convert annotated frame matrix back to ROS image format
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to translate image arrays: {e}')

        # Log complete pipeline performance costs
        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        
        self.get_logger().info(
            f"\n--- ONNX CAMERA PERFORMANCE PROFILE ---\n"
            f"Input Frame Resolution: {w_orig}x{h_orig}\n"
            f"Detected Objects      : {len(detections.detections)}\n"
            f"Pre-Callback Age      : {data_age_ms:.2f}ms\n"
            f"Pure ONNX Inference   : {inference_time_ms:.2f}ms\n"
            f"Total Callback Cost   : {node_time_ms:.2f}ms"
        )
    def save_latency_distribution_plot(self):
        """ METHOD 3: Saves a latency histogram plot inside the dynamic package folder """
        if not self.total_latencies:
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(self.inference_latencies, bins=20, alpha=0.6, label='Pure YOLO Inference Time', color='g')
        plt.hist(self.total_latencies, bins=20, alpha=0.6, label='Total Callback Cost', color='b')
        plt.axvline(np.mean(self.total_latencies), color='r', linestyle='dashed', linewidth=2, 
                    label=f'Mean Callback ({np.mean(self.total_latencies):.2f}ms)')
        
        plt.title('Camera Node Latency Distribution Over Time')
        plt.xlabel('Execution Latency (milliseconds)')
        plt.ylabel('Frequency (Frames)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Assemble path: install/kitti_r1_perception/share/kitti_r1_perception/perf_logs/camera_latency_distribution.png
        log_dir = os.path.join(self.package_share_dir, 'perf_logs')
        os.makedirs(log_dir, exist_ok=True)
        plot_path = os.path.join(log_dir, 'camera_latency_distribution.png')
        
        plt.savefig(plot_path)
        plt.close()
        self.get_logger().info(f"\n[PERF DATA LOG] Latency distribution graph successfully saved to: {plot_path}\n")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CameraDetectorNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.save_latency_distribution_plot()
        node.destroy_node()

        # FIXED: Enforce clear context protection guard for ROS 2 Jazzy
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

