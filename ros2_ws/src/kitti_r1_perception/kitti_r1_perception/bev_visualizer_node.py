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

# --- Detection styles ---
# Class-specific BGR colors used to render fused detections.
CLASS_VISUALS = {
    0: {"name": "Car",        "color": (0, 0, 255)},    # Red
    1: {"name": "Pedestrian", "color": (255, 0, 0)},    # Blue
    2: {"name": "Cyclist",    "color": (0, 255, 255)},  # Yellow
    3: {"name": "Truck",      "color": (0, 128, 255)},  # Orange
    4: {"name": "Bus",        "color": (255, 0, 255)},  # Magenta
    5: {"name": "Motorcycle", "color": (255, 255, 0)},  # Cyan
    6: {"name": "Traffic Sign", "color": (128, 0, 128)},  # Purple
    7: {"name": "Traffic Light", "color": (0, 255, 0)},  # Green
    8: {"name": "Barrier",    "color": (128, 128, 0)},  # Olive
    9: {"name": "Animal",     "color": (0, 128, 128)},  # Teal
    10: {"name": "Other",      "color": (128, 128, 128)}  # Gray
}


class BEVVisualizerNode(Node):
    def __init__(self):
        super().__init__('bev_visualizer_node')
        
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        
        # --- Configuration ---
        # Expose BEV bounds and resolution as launch parameters.
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

        # --- Frame-rate tracking ---
        self.last_frame_time = None
        self.current_fps = 0.0
        self.alpha_fps = 0.1  # Higher values respond faster; lower values are smoother.

        # --- Input subscriptions ---
        # Keep the latest detections for rendering with each point-cloud frame.
        self.latest_fused_msg = None

        bev_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,     
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Keep the queue short to limit visualization lag.
        )
        
        # Subscribe to fused detections rather than raw LiDAR detections.
        self.fused_sub = self.create_subscription(
            Detection3DArray, '/detection/fused_3d', self.fused_callback, bev_qos
        )
        # Use the ground-filtered cloud published by the LiDAR processing node.
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
        """ Asynchronous Cache: Instantly updates object references in RAM."""
        self.latest_fused_msg = msg

    def no_ground_callback(self, msg):
        """ Processing Trigger: Executed by the lean, pre-filtered cloud topic."""
        callback_start = time.perf_counter()
        fused_msg = self.latest_fused_msg

        # --- Point-cloud conversion ---
        # View the message buffer directly to avoid copying point data.
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        num_points = msg.width * msg.height
        if num_points == 0:
            return
            
        points_struct = raw_data.view(dtype=np.float32).reshape(num_points, -1)
        points = points_struct[:, :3]

        # Map world coordinates to BEV pixel coordinates.
        x_pix = ((self.max_y - points[:, 1]) / self.resolution).astype(np.int32)
        y_pix = ((self.max_x - points[:, 0]) / self.resolution).astype(np.int32)

        # Initialize the BEV image canvas.
        bev_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Render in-bounds LiDAR points in gray using flat pixel indices.
        valid_mask = (x_pix >= 0) & (x_pix < self.width) & (y_pix >= 0) & (y_pix < self.height)
        if np.any(valid_mask):
            flat_indices = y_pix[valid_mask] * self.width + x_pix[valid_mask]
            bev_img.reshape(-1, 3)[flat_indices] = [180, 180, 180]

        # --- BEV grid and detections ---
        # Map the sensor origin to the BEV image coordinate system.
        origin_px_x = int((self.max_y - 0.0) / self.resolution)
        origin_px_y = int((self.max_x - 0.0) / self.resolution)
        
        # Draw the metric grid for spatial reference.
        self._draw_metric_grid(bev_img, origin_px_x, origin_px_y)

        # Render the latest fused detections when available.
        if fused_msg is not None:
            self._render_fused_detections(fused_msg, bev_img)

        # Measure rendering time for the diagnostic overlay.
        render_cost_ms = (time.perf_counter() - callback_start) * 1000.0

        # Update the exponentially smoothed frame rate.
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            delta_time = current_time - self.last_frame_time
            if delta_time > 0.0:
                instantaneous_fps = 1.0 / delta_time
                self.current_fps = (self.alpha_fps * instantaneous_fps) + ((1.0 - self.alpha_fps) * self.current_fps)
        self.last_frame_time = current_time

        # --- Diagnostic overlay ---
        # Add a dark background to improve text contrast.
        cv2.rectangle(bev_img, (5, 5), (145, 45), (15, 15, 18), -1)        
        # Display the smoothed frame rate and render time.
        cv2.putText(bev_img, f"PIPELINE: {self.current_fps:.1f} FPS", (12, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bev_img, f"BEV COST: {render_cost_ms:.1f} ms", (12, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Publish the rendered BEV image.
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = msg.header
        self.bev_pub.publish(bev_msg)
        
        total_cost_ms = (time.perf_counter() - callback_start) * 1000.0
        self.get_logger().info(
            f"Asynchronous BEV Render Complete. Render: {render_cost_ms:.2f}ms | Total Pipeline Callback: {total_cost_ms:.2f}ms", 
            throttle_duration_sec=2.0
        )

    def _render_fused_detections(self, fused_msg, bev_img):
        # --- Fused detections ---
        for det in fused_msg.detections:
            cx = float(det.bbox.center.position.x)  
            cy = float(det.bbox.center.position.y)
            sx = float(det.bbox.size.x)  # Object length along the x-axis.
            sy = float(det.bbox.size.y)  # Object width along the y-axis.

            # Skip detections whose centers fall outside the configured BEV bounds.
            if not (self.min_x <= cx <= self.max_x and self.min_y <= cy <= self.max_y):
                continue
                    
            distance_meters = np.hypot(cx, cy)
                
            # Convert the orientation quaternion to a yaw angle.
            q = det.bbox.center.orientation
            _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])

            # Define the box corners in the local KITTI ground plane.
            # The x-axis points forward and the y-axis points left.
                 
            # Order the corners clockwise, starting at the front-left.
            local_corners = np.array([
                    [ sx/2,  sy/2],  # Front-left
                    [ sx/2, -sy/2],  # Front-right
                    [-sx/2, -sy/2],  # Rear-right
                    [-sx/2,  sy/2]   # Rear-left
                ], dtype=np.float32)

            # Build a 2D rotation matrix from the yaw angle.
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[cos_y, -sin_y],
                                [sin_y,  cos_y]], dtype=np.float32)

            # Rotate the corners and translate them into world coordinates.
            world_corners = np.dot(local_corners, rot_mat.T) + np.array([cx, cy], dtype=np.float32)

            # Map the transformed corners to BEV pixel coordinates.
            pix_corners = np.zeros_like(world_corners, dtype=np.int32)
            pix_corners[:, 0] = ((self.max_y - world_corners[:, 1]) / self.resolution).astype(np.int32) 
            pix_corners[:, 1] = ((self.max_x - world_corners[:, 0]) / self.resolution).astype(np.int32) 
                
            # Extract the detection class ID.
            cls_id = self._parse_class_id(det)                    

            # Use the class-specific style, or a green fallback for unknown classes.
            visual = CLASS_VISUALS.get(cls_id, {"name": f"ID:{cls_id}", "color": (0, 255, 0)})
                
            # Draw the projected bounding-box outline.
            cv2.polylines(bev_img, [pix_corners.reshape(-1, 1, 2)], isClosed=True, color=visual["color"], thickness=2)
                
            f_mid_px, f_mid_py, center_pixel_x, center_pixel_y = self._draw_heading_vector(bev_img, cx, cy, sx, rot_mat, visual)

            self._log_heading_verification(distance_meters, q, yaw, cls_id, f_mid_px, f_mid_py, center_pixel_x, center_pixel_y)

            # Label the detection with its class and distance.
            display_text = f"{visual['name']} [{distance_meters:.1f}m]"
            cv2.putText(bev_img, display_text, (int(pix_corners[0, 0]), int(pix_corners[0, 1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, visual["color"], 1, cv2.LINE_AA)

    def _log_heading_verification(self, distance_meters, q, yaw, cls_id, f_mid_px, f_mid_py, c_px, c_py):
        # Log heading geometry at debug level for optional diagnostics.
        dx = f_mid_px - c_px
        dy = f_mid_py - c_py
        self.get_logger().debug(
                    f"\n[HEADING CHECK] Object Class: {cls_id} | Distance: {distance_meters:.1f}m\n"
                    f"└─ Raw Yaw (Radians): {yaw:.4f} | Quat [w,x,y,z]: [{q.w:.2f}, {q.x:.2f}, {q.y:.2f}, {q.z:.2f}]\n"
                    f"└─ Pixel Delta vector [dX, dY]: [{dx}, {dy}]"
                )

    def _draw_heading_vector(self, bev_img, cx, cy, sx, rot_mat, visual):
        # Transform the front-center point into world coordinates.
        front_mid_world = np.dot(np.array([sx/2, 0.0], dtype=np.float32), rot_mat.T) + np.array([cx, cy], dtype=np.float32)
        # Map the front-center point to BEV pixel coordinates.
        f_mid_px = int((self.max_y - front_mid_world[1]) / self.resolution)
        f_mid_py = int((self.max_x - front_mid_world[0]) / self.resolution)
        # Map the object center to BEV pixel coordinates.
        center_pixel_x = int((self.max_y - cy) / self.resolution)  # World y maps to pixel x.
        center_pixel_y = int((self.max_x - cx) / self.resolution)  # World x maps to pixel y.
        
        # Draw a line from the object center to its front center to indicate heading.
        cv2.line(bev_img, (center_pixel_x, center_pixel_y), (f_mid_px, f_mid_py), visual["color"], 2)
        return f_mid_px,f_mid_py,center_pixel_x,center_pixel_y

    def _parse_class_id(self, det):
        """Extracts the class ID from a Detection3D message, handling potential variations in ROS 2 message structures."""
        if det.results and len(det.results) > 0:
            first_result = det.results[0]
            if hasattr(first_result, 'hypothesis'):
                return int(first_result.hypothesis.class_id)
            else:
                return int(first_result.class_id)
        return -1  # Use -1 when the detection has no class ID.

    def _draw_metric_grid(self, bev_img, origin_px_x, origin_px_y):
        # --- Distance grid ---
        # Draw range rings at fixed distances from the sensor.
        ring_distances_meters = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for dist in ring_distances_meters:
            # Convert the ring radius from meters to pixels.
            radius_px = int(dist / self.resolution)
            # Draw the range ring.
            cv2.circle(bev_img, (origin_px_x, origin_px_y), radius_px, (60, 60, 60), thickness=1, lineType=cv2.LINE_AA)
            
            # Place the distance label along the forward centerline.
            text_y_px = int((self.max_x - dist) / self.resolution)
            # Skip labels outside the visible image.
            if 0 <= text_y_px < self.height:
                # Draw a dark outline to keep the label readable over point clouds.
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                # Draw the foreground label.
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw the forward centerline, which represents world y = 0.
        cv2.line(bev_img, (origin_px_x, 0), (origin_px_x, self.height), (45, 45, 48), thickness=1)

        # Draw lateral grid lines at 10-meter intervals.
        for lateral_meter in [-20.0, -10.0, 10.0, 20.0]:
            lat_px_x = int((self.max_y - lateral_meter) / self.resolution)
            if 0 <= lat_px_x < self.width:
                cv2.line(bev_img, (lat_px_x, 0), (lat_px_x, self.height), (35, 35, 38), thickness=1, lineType=cv2.LINE_4)

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
