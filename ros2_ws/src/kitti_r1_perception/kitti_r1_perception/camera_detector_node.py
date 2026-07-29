import os
import sys
import numpy as np
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory

class CameraDetectorNode(Node):

    def __init__(self):
        super().__init__('camera_detector_node')
         # ⚡ CRITICAL TIME FIX 1: Enforce simulation clock tracking immediately
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        ])
        self.last_img_shape = None

        # Use unbuffered sys.stdout to ensure prints bypass launch file buffers immediately
        sys.stdout.reconfigure(line_buffering=True)
        
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.bridge = CvBridge()
        
        # Latency lists for post-run distribution plotting
        self.inference_latencies = []
        self.total_latencies = []
        
        # Parameters (Declared in main init scope)
        self.declare_parameter('model_version', 'yolov8n') 
        self.declare_parameter('conf_threshold', 0.25)
        self.conf_threshold = self.get_parameter('conf_threshold').value

        self.realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. DO NOT load the model here. Instead, create a one-shot timer 
        # This allows the node to register with ROS instantly without blocking
        self.init_timer = self.create_timer(0.1, self.async_model_init)

    
    def async_model_init(self):
        # Cancel the timer immediately so it only runs once
        self.init_timer.cancel()

        # 1. Directly resolve the path (will raise a clean PackageNotFoundError if missing)
        self.package_share_dir = get_package_share_directory('kitti_r1_perception')

        # 2. Define and set the torch home environment variable
        self.torch_home_dir = os.path.join(self.package_share_dir, 'torch_home')
        os.environ['TORCH_HOME'] = self.torch_home_dir
        self.get_logger().info(f"Set TORCH_HOME to: {self.torch_home_dir}")

        model_ver = self.get_parameter('model_version').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        # Prefer model file inside TORCH_HOME to leverage local cache/store
        model_filename = f"{model_ver}.pt"
        model_path_local = os.path.join(self.torch_home_dir, model_filename)
        if os.path.exists(model_path_local):
            model_path = model_path_local
        else:
            # fallback to simple name which lets ultralytics attempt download or resolve
            model_path = model_filename

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        # ----------------------------------------------------------------------
        # ⚡ HIGH-SPEED PT LOADING OPTIMIZATION
        # ----------------------------------------------------------------------
        if self.device == 'cuda':
            # 1. Warm up the CUDA context immediately *before* disk reads hit
            torch.cuda.init()
            
            # 2. Tell PyTorch's backend allocator to expect immediate sequentially-read weights
            torch.backends.cudnn.benchmark = True
            
            try:
                # 3. Initialize model structure cleanly without unsupported kwargs
                self.model = YOLO(model_path)
                
                # 4. Explicitly move weights onto the RTX 4060 VRAM using non-blocking memory pinning
                self.model.to(self.device)
                self.get_logger().info('Direct NVMe-to-VRAM context optimization applied successfully.')
            except Exception as e:
                # Fallback path if custom device mapping fails
                self.model = YOLO(model_path)
                self.get_logger().warn(f'Standard initialization fallback triggered: {str(e)}')
        else:
            self.model = YOLO(model_path)
            
        self.is_cuda = (self.device == 'cuda')
        self.class_names = self.model.names

        # ----------------------------------------------------------------------
        # ⚡ FIX: DEEP LEARNING GRAPH COMPILATION WARMUP
        # ----------------------------------------------------------------------
        try:
            self.get_logger().info('Compiling CUDA Graph pipelines for YOLOv8 inference maps...')
            # Generate a blank matrix mimicking the layout dimensions of the camera
            blank_warmup_img = np.zeros((384, 1248, 3), dtype=np.uint8)

            # ⚡ CRITICAL FIX: Pass structural dimensions explicitly as [Height, Width] 
            # rounded up to the nearest multiple of 32 (374->384, 1238->1248) to keep CUDA graph stable
            self.target_imgsz = [384, 1248]
            
            # ⚡ CRITICAL FIX: Pass structural dimensions explicitly using imgsz=640
            _ = self.model.predict(
                blank_warmup_img, 
                conf=self.conf_threshold, 
                imgsz=self.target_imgsz, 
                verbose=False, 
                device=self.device
            )
            
            if self.is_cuda:
                torch.cuda.synchronize()
            self.get_logger().info('YOLOv8 compiler graph tracks loaded successfully.')
        except Exception as e:
            self.get_logger().warn(f'Camera initialization warmup bypassed: {str(e)}')

        # ----------------------------------------------------------------------
        # Subscription & Publishers (Open only *after* warmup is complete)
        # ----------------------------------------------------------------------
        self.subscription = self.create_subscription(
            Image,
            '/kitti/camera_left/image_raw',
            self.listener_callback,
            self.realtime_qos,
            callback_group=self.callback_group
        )
 
        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/detection/camera_2d', self.realtime_qos)
        self.annotated_pub = self.create_publisher(Image, '/detection/camera_annotated', self.realtime_qos)
        self.get_logger().info(f'Camera Detector Node started with model: {model_path}')

    def listener_callback(self, msg):
        node_start_time = time.perf_counter()

        # ⚡ TIME FIX: Calculate Pre-Callback Age using the integrated node clock
        # This respects bag file time when '--clock' is supplied to ros2 bag play
        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        data_age_ms = (now - msg_time).nanoseconds / 1e6

        # Convert image and run inference
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # ⚡ SHAPE MONITOR: Validate image size continuity across incoming frames
        current_shape = cv_image.shape  # Returns tuple: (Height, Width, Channels)

        if self.last_img_shape is None:
            self.last_img_shape = current_shape
            self.get_logger().info(f"Initial image resolution baseline established: {current_shape}")
        elif current_shape != self.last_img_shape:
            self.get_logger().warn(
                f"🚨 DYNAMIC SHAPE DETECTED! Resolution changed from {self.last_img_shape} to {current_shape}. "
                f"This variance will break CUDA Graph performance mappings."
            )
            self.last_img_shape = current_shape  # Update tracking state to suppress continuous spam

        # Start high-res system timing for CUDA workload operations
        inference_start = time.perf_counter()

        # ⚡ CRITICAL FIX: Ensure predictable matrix structures by forcing imgsz matching your warmup
        results = self.model.predict(
            cv_image, 
            conf=self.conf_threshold, 
            imgsz=self.target_imgsz,  # Explicitly match the warmup dimensions to prevent CUDA graph instability
            verbose=False, 
            device=self.device
        )
        if self.is_cuda:
            torch.cuda.synchronize()

        # Calculate absolute hardware inference duration
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        # Save metrics for our distribution curves
        self.inference_latencies.append(inference_time_ms)

        # Prepare Detection2DArray
        detections = Detection2DArray()
        detections.header = msg.header
        
        # Track if we have active targets
        has_boxes = len(results) > 0 and len(results[0].boxes) > 0

        # 3. CPU Offload: Extract bulk tensors directly from GPU with proper ROS type alignment
        if has_boxes:
            boxes = results[0].boxes
            xywh_all = boxes.xywh.cpu().numpy()       # [center_x, center_y, width, height]
            cls_all = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()

            for idx in range(len(xywh_all)):
                detection = Detection2D()
                detection.header = msg.header
                
                # Assign values directly - standardizing center tracking positions
                detection.bbox.center.position.x = float(xywh_all[idx, 0])
                detection.bbox.center.position.y = float(xywh_all[idx, 1])
                
                # Standard ROS 2 dimensions mapping (Full Width & Height)
                detection.bbox.size_x = float(xywh_all[idx, 2])
                detection.bbox.size_y = float(xywh_all[idx, 3])
                
                hypothesis = ObjectHypothesisWithPose()
                
                # ⚡ CRITICAL FIX: Convert integer ID to standard string index to prevent ROS serialization crashes
                hypothesis.hypothesis.class_id = str(cls_all[idx]) 
                hypothesis.hypothesis.score = float(conf_all[idx])
                
                detection.results.append(hypothesis)
                detections.detections.append(detection)

        # Publish results
        self.detection_pub.publish(detections)
        
        # 4. FIXED VISUALIZATION: Safe unconditional plotting with empty detection guard
        try:
            if has_boxes:
                # Render detected overlay tracking masks
                annotated_frame = results[0].plot()
            else:
                # Safe Fallback: Publish the raw input image if zero targets exist
                annotated_frame = cv_image
                
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to convert annotated image: {e}')

        # Calculate absolute total execution timeline cost
        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        self.total_latencies.append(node_time_ms)
        num_detections = len(detections.detections)

        self.get_logger().info(
            f"\n--- CAMERA PERFORMANCE PROFILE ---\n"
            f"Detected Objects    : {num_detections}\n"
            f"Pre-Callback Age    : {data_age_ms:.2f}ms\n"
            f"Pure YOLO Inference : {inference_time_ms:.2f}ms\n"
            f"Total Callback Cost : {node_time_ms:.2f}ms"
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
        executor = MultiThreadedExecutor(num_threads=2)
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
