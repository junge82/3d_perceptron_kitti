import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d
import time

# 5. Draw Fused Detections with Distinct Class Styling
CLASS_VISUALS = {
    0: {"name": "Car",        "color": (0, 0, 255)},   # Bright Red Box
    1: {"name": "Pedestrian", "color": (255, 0, 0)},   # Deep Blue Box
    2: {"name": "Cyclist",    "color": (0, 255, 255)},  # Neon Yellow Box
    3: {"name": "Truck",      "color": (0, 128, 255)},   # Orange Box
    4: {"name": "Bus",        "color": (255, 0, 255)},   # Magenta Box
    5: {"name": "Motorcycle", "color": (255, 255, 0)},   # Cyan Box
    6: {"name": "Traffic Sign", "color": (128, 0, 128)},   # Purple Box
    7: {"name": "Traffic Light", "color": (0, 255, 0)},   # Green Box
    8: {"name": "Barrier",    "color": (128, 128, 0)},   # Olive Box
    9: {"name": "Animal",     "color": (0, 128, 128)},   # Teal Box
    10: {"name": "Other",      "color": (128, 128, 128)}  # Gray Box
}


class BEVVisualizerNode(Node):
    def __init__(self):
        super().__init__('bev_visualizer_node')
        
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        
        # Parameters exposed to launch configuration for easy tuning
        self.declare_parameter('bev_min_x', 1.5)
        self.declare_parameter('bev_max_x', 50.0)
        self.declare_parameter('bev_min_y', -25.0)
        self.declare_parameter('bev_max_y', 25.0)
        self.declare_parameter('bev_resolution', 0.10)

        self.min_x = self.get_parameter('bev_min_x').value
        self.max_x = self.get_parameter('bev_max_x').value
        self.min_y = self.get_parameter('bev_min_y').value
        self.max_y = self.get_parameter('bev_max_y').value
        self.resolution = self.get_parameter('bev_resolution').value

        self.width = int((self.max_y - self.min_y) / self.resolution)
        self.height = int((self.max_x - self.min_x) / self.resolution)
        self.bridge = CvBridge()

        # ⚡ NEW: Frame timing variables for live FPS tracking
        self.last_frame_time = None
        self.current_fps = 0.0
        self.alpha_fps = 0.1  # Smoothing factor (higher = faster update, lower = smoother)

        # ⚡ 1. MEMORY CACHE POINTER FOR TRACKED OBJECTS
        self.latest_fused_msg = None

        bev_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,     
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Drop old messages instantly to prevent lag accumulation
        )
        
        # ⚡ 2. SUBSCRIBE ONLY TO FINISHED PIPELINE OUTPUTS (NO RAW LIDAR)
        self.fused_sub = self.create_subscription(
            Detection3DArray, '/detection/fused_3d', self.fused_callback, bev_qos
        )
        # Driven by the pre-segmented groundless cloud from your Lidar cluster node
        self.no_ground_sub = self.create_subscription(
            PointCloud2, '/lidar/points_no_ground', self.no_ground_callback, bev_qos
        )

        bev_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.bev_pub = self.create_publisher(Image, '/visualization/bev_image', bev_pub_qos)
        self.get_logger().info(f'Lightweight Asynchronous BEV Visualizer Started: {self.width}x{self.height}')

    def fused_callback(self, msg):
        """⚡ Asynchronous Cache: Instantly updates object references in RAM."""
        self.latest_fused_msg = msg

    def no_ground_callback(self, msg):
        """⚡ Processing Trigger: Executed by the lean, pre-filtered cloud topic."""
        callback_start = time.perf_counter()
        fused_msg = self.latest_fused_msg

        # Direct memory casting of your clean X, Y, Z coordinates (<0.2ms)
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        num_points = msg.width * msg.height
        if num_points == 0:
            return
            
        points_struct = raw_data.view(dtype=np.float32).reshape(num_points, -1)
        points = points_struct[:, :3]

        # Project world coordinates into aligned pixel positions
        x_pix = ((self.max_y - points[:, 1]) / self.resolution).astype(np.int32)
        y_pix = ((self.max_x - points[:, 0]) / self.resolution).astype(np.int32)

        # Generate base image canvas
        bev_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Paint Lidar obstacle points as gray using direct 1D memory array masking
        valid_mask = (x_pix >= 0) & (x_pix < self.width) & (y_pix >= 0) & (y_pix < self.height)
        if np.any(valid_mask):
            flat_indices = y_pix[valid_mask] * self.width + x_pix[valid_mask]
            bev_img.reshape(-1, 3)[flat_indices] = [180, 180, 180]

        # --- DRAW CONCENTRIC DISTANCE RINGS ---
        # Real-world sensor origin (0,0) projected into our asymmetric pixel coordinate frame
        origin_px_x = int((self.max_y - 0.0) / self.resolution)
        origin_px_y = int((self.max_x - 0.0) / self.resolution)
        
        # Target distance thresholds in meters
        ring_distances_meters = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for dist in ring_distances_meters:
            # Convert real-world meter radius directly to pixel radius
            radius_px = int(dist / self.resolution)
            # Draw distance circles
            cv2.circle(bev_img, (origin_px_x, origin_px_y), radius_px, (60, 60, 60), thickness=1, lineType=cv2.LINE_AA)
            
            # Calculate text drawing anchor point (placed along the center forward centerline)
            text_y_px = int((self.max_x - dist) / self.resolution)
            # Only draw the text label if it falls inside the visible image canvas boundary
            if 0 <= text_y_px < self.height:
                # Add text backing outline shadow for high visibility against dense point clouds
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                # Clear white text overlay
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # ⚡ FIXED: DRAW METRIC GRID ANCHORS EXACTLY ONCE (OUTSIDE THE LOOP)
        # Draw forward centerline (Y = 0 in world space) points straight up the middle
        cv2.line(bev_img, (origin_px_x, 0), (origin_px_x, self.height), (45, 45, 48), thickness=1)

        # Draw lateral metric grid lines every 10 meters on the sides
        for lateral_meter in [-20.0, -10.0, 10.0, 20.0]:
            lat_px_x = int((self.max_y - lateral_meter) / self.resolution)
            if 0 <= lat_px_x < self.width:
                cv2.line(bev_img, (lat_px_x, 0), (lat_px_x, self.height), (35, 35, 38), thickness=1, lineType=cv2.LINE_4)

        if fused_msg is not None:
            for det in fused_msg.detections:
                cx = float(det.bbox.center.position.x)  
                cy = float(det.bbox.center.position.y)
                sx = float(det.bbox.size.x)     # Length (x-axis)
                sy = float(det.bbox.size.y)     # Width (y-axis)

                # Skip drawing immediately if the object center is outside the operational cluster ROI
                if not (self.min_x <= cx <= self.max_x and self.min_y <= cy <= self.max_y):
                    continue
                    
                distance_meters = np.hypot(cx, cy)
                
                # Unpack quaternion layout to calculate object yaw heading rotation angle
                q = det.bbox.center.orientation
                _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])

                # Generate the bounding box perimeter corners relative to its local center coordinate
                # In the standard KITTI coordinate frame, 
                # the tracking definitions are mapped using the standard right-hand rule system:
                # X-axis points directly Forward (Longitudinal axis).
                # Y-axis points directly Left (Lateral axis).
                 
                # 1. Local 2D corner bounding matrix tracking coordinates
                local_corners = np.array([
                    [ sx/2,  sy/2],  # Front-Left
                    [ sx/2, -sy/2],  # Front-Right
                    [-sx/2, -sy/2],  # Rear-Right
                    [-sx/2,  sy/2]   # Rear-Left
                ], dtype=np.float32)

                # Build 2D rotation matrix using computed yaw
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                rot_mat = np.array([[cos_y, -sin_y],
                                     [sin_y,  cos_y]], dtype=np.float32)

                # Vectorized transform to world coordinates (Fixed row-by-column math order)
                world_corners = np.dot(local_corners, rot_mat.T) + np.array([cx, cy], dtype=np.float32)

                # Project transformed corners into the aligned asymmetric pixel space
                pix_corners = np.zeros_like(world_corners, dtype=np.int32)
                pix_corners[:, 0] = ((self.max_y - world_corners[:, 1]) / self.resolution).astype(np.int32) 
                pix_corners[:, 1] = ((self.max_x - world_corners[:, 0]) / self.resolution).astype(np.int32) 
                
                # ROS 2 Jazzy Attribute Reading
                cls_id = -1
                if det.results and len(det.results) > 0:
                    first_result = det.results[0]
                    if hasattr(first_result, 'hypothesis'):
                        cls_id = int(first_result.hypothesis.class_id)
                    else:
                        cls_id = int(first_result.class_id)

                visual = CLASS_VISUALS.get(cls_id, {"name": f"ID:{cls_id}", "color": (0, 255, 0)})
                
                # Draw rotated box bounding shape with matching opencv tensor rules
                cv2.polylines(bev_img, [pix_corners.reshape(-1, 1, 2)], isClosed=True, color=visual["color"], thickness=2)
                
                # Draw heading line indicators
                front_mid_world = np.dot(np.array([sx/2, 0.0], dtype=np.float32), rot_mat.T) + np.array([cx, cy], dtype=np.float32)
                f_mid_px = int((self.max_y - front_mid_world[1]) / self.resolution)
                f_mid_py = int((self.max_x - front_mid_world[0]) / self.resolution)
                c_px = int((self.max_y - cy) / self.resolution)
                c_py = int((self.max_x - cx) / self.resolution)
                cv2.line(bev_img, (c_px, c_py), (f_mid_px, f_mid_py), visual["color"], 2)

                # ⚡ SILENT BACKGROUND HEADING VERIFICATION
                dx = f_mid_px - c_px
                dy = f_mid_py - c_py
                self.get_logger().debug(
                    f"\n[HEADING CHECK] Object Class: {cls_id} | Distance: {distance_meters:.1f}m\n"
                    f"└─ Raw Yaw (Radians): {yaw:.4f} | Quat [w,x,y,z]: [{q.w:.2f}, {q.x:.2f}, {q.y:.2f}, {q.z:.2f}]\n"
                    f"└─ Pixel Delta vector [dX, dY]: [{dx}, {dy}]"
                )

                # Overlay distance tag text string properties
                display_text = f"{visual['name']} [{distance_meters:.1f}m]"
                cv2.putText(bev_img, display_text, (int(pix_corners[0, 0]), int(pix_corners[0, 1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, visual["color"], 1, cv2.LINE_AA)

        # Calculate pure execution latency before drawing the tracker panel
        cost_ms = (time.perf_counter() - callback_start) * 1000.0

        # ⚡ COMPUTE LIVE EXPONENTIAL MOVING AVERAGE PIPELINE FPS
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            delta_time = current_time - self.last_frame_time
            if delta_time > 0.0:
                instantaneous_fps = 1.0 / delta_time
                self.current_fps = (self.alpha_fps * instantaneous_fps) + ((1.0 - self.alpha_fps) * self.current_fps)
        self.last_frame_time = current_time

        # ⚡ DRAW DIAGNOSTIC OVERLAY PANEL (Top-Left Dashboard)
        # Draws a dark background backing rectangle for perfect contrast
        cv2.rectangle(bev_img, (5, 5), (145, 45), (15, 15, 18), -1)
        
        # Overlay running FPS (Neon Green) and node render cost (Light Gray)
        cv2.putText(bev_img, f"PIPELINE: {self.current_fps:.1f} FPS", (12, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bev_img, f"BEV COST: {cost_ms:.1f} ms", (12, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # 6. Publish processed visualization frame
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = msg.header
        self.bev_pub.publish(bev_msg)
        
        cost_ms = (time.perf_counter() - callback_start) * 1000.0
        self.get_logger().info(f"⚡ Asynchronous BEV Render Complete. Cost: {cost_ms:.2f}ms", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = BEVVisualizerNode()
    executor = SingleThreadedExecutor()
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
