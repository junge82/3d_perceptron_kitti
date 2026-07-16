import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # Zero-Backlog QoSProfile
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
import time  # High-precision CPU timer for profiling

class CameraDetectorNode(Node):
    def __init__(self):
        super().__init__('camera_detector_node')
        
        # 1. Create a Reentrant Callback Group
        # This allows multiple callbacks in this node to run concurrently on different threads
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # Parameters
        self.declare_parameter('model_version', 'yolov8n') # Options: yolov8n, yolo11n
        self.declare_parameter('conf_threshold', 0.25)
        
        model_ver = self.get_parameter('model_version').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        
        # Load model (downloads if not present)
        model_path = f"{model_ver}.pt"
        # Choose device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        
        # Check if true CUDA sync timing is needed
        self.is_cuda = (self.device == 'cuda')
        
        # Move model to device if supported
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.class_names = self.model.names
        
        self.bridge = CvBridge()

        # Create a real-time, low-latency QoS configuration
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Don't block for missing frames
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # IMMEDIATELY drop old images if the callback is busy
        )
        
        # Subscription
        self.subscription = self.create_subscription(
            Image,
            '/kitti/camera_left/image_raw',
            self.listener_callback,
            realtime_qos,
            callback_group=self.callback_group # CRITICAL: Ensures callback isolation for multi-threaded execution
        )
 
        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/detection/camera_2d', realtime_qos)
        self.annotated_pub = self.create_publisher(Image, '/detection/camera_annotated', realtime_qos)
        self.get_logger().info(f'Camera Detector Node started with model: {model_path}')

    def listener_callback(self, msg):
        # Start total callback latency timer
        node_start_time = time.perf_counter()

        # Extract current simulation timeline clock safely using ROS_TIME configuration (FIXED .stamp layout)
        now = self.get_clock().now()
        msg_stamp = rclpy.time.Time.from_msg(msg.header.stamp, clock_type=rclpy.time.ClockType.ROS_TIME)
        
        # Calculate pre-existing data age before execution starts
        data_age_ms = (now - msg_stamp).nanoseconds / 1e6

        # Live-update parameters from the ROS 2 server
        self.conf_threshold = self.get_parameter('conf_threshold').value

        # 1. Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # 2. Run Inference (Profile precise execution isolation)
        inference_start = time.perf_counter()
        results = self.model.predict(cv_image, conf=self.conf_threshold, verbose=False, device=self.device)
        
        if self.is_cuda:
            torch.cuda.synchronize()  # Force execution pipeline finish to avoid asynchronous skewing
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        # Prepare Detection2DArray
        detections = Detection2DArray()
        detections.header = msg.header
        
        for result in results:
            for box in result.boxes:
                # Extract box coordinates (FIXED raw tensor access safely)
                xywh = box.xywh[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                
                detection = Detection2D()
                detection.header = msg.header
                
                # Bounding box
                detection.bbox.center.position.x = float(xywh[0])
                detection.bbox.center.position.y = float(xywh[1])
                detection.bbox.size_x = float(xywh[2])
                detection.bbox.size_y = float(xywh[3])
                
                # Hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = self.class_names[cls_id]
                hypothesis.hypothesis.score = conf
                detection.results.append(hypothesis)
                
                detections.detections.append(detection)

        # Publish results
        self.detection_pub.publish(detections)
        
        # --- Visualize and Publish the results (FIXED array references) ---
        if len(results) > 0:
            try:
                annotated_frame = results[0].plot()
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to convert and publish annotated image: {e}')

        # Finish total node computation metric tracking
        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        num_detections = len(detections.detections)

        # Output detailed execution performance layout
        self.get_logger().info(
            f"\n--- CAMERA PERFORMANCE PROFILE ---\n"
            f"Detected Objects    : {num_detections}\n"
            f"Pre-Callback Age    : {data_age_ms:.2f}ms\n"
            f"Pure YOLO Inference : {inference_time_ms:.2f}ms\n"
            f"Total Callback Cost : {node_time_ms:.2f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetectorNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
