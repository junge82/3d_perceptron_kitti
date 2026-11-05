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
        """
        Processing Trigger Orchestrator: Coordinates raw sensor telemetry parsing, 
        matrix projection, visual layer rendering, and ROS publication.
        """
        callback_start = time.perf_counter()
        
        # 1. Parse raw byte streams into structured numerical matrices
        points = self._parse_point_cloud(msg)
        if points is None:
            return

        # 2. Build canvas layer and project point distributions via Affine Matrix
        bev_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._project_cloud_to_bev(points, bev_img)

        # 3. Layer contextual geometric features onto the frame
        self._render_spatial_grid_layers(bev_img)
        
        if self.latest_fused_msg is not None:
            self._render_fused_detections(self.latest_fused_msg, bev_img)

        # 4. Process streaming frame-rate telemetry and overlay instrumentation
        render_cost_ms = (time.perf_counter() - callback_start) * 1000.0
        self._update_pipeline_fps()
        self._draw_diagnostic_dashboard(bev_img, render_cost_ms)

        # 5. Package frame data definitions and publish to the ROS graph
        self._publish_bev_frame(bev_img, msg.header, callback_start)

    def _parse_point_cloud(self, msg):
        """Zero-copy buffer parser mapping input data streams to coordinate matrices."""
        num_points = msg.width * msg.height
        if num_points == 0:
            return None
            
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        points_struct = raw_data.view(dtype=np.float32).reshape(num_points, -1)
        return points_struct[:, :3]

    def _project_cloud_to_bev(self, points, bev_img):
        """
        Applies a unified 2x3 Affine Transformation matrix to project 
        homogeneous 3D point cloud layers onto the 2D image plane.
        """
        # Construct homogeneous space matrices [N, 3] -> [X_lidar, Y_lidar, 1]
        homogeneous_pts = np.hstack([points[:, :2], np.ones((points.shape[0], 1), dtype=np.float32)])

        # Use the shared projection transform with no local rotation or translation.
        M_proj = self._build_projection_matrix()

        # Vectorized project matrix calculation via optimized inner product dot notation
        pix_coords = np.dot(M_proj, homogeneous_pts.T).T.astype(np.int32)
        x_pix, y_pix = pix_coords[:, 0], pix_coords[:, 1]

        # Apply boundary masking and update visual canvas states
        valid_mask = (x_pix >= 0) & (x_pix < self.width) & (y_pix >= 0) & (y_pix < self.height)
        if np.any(valid_mask):
            flat_indices = y_pix[valid_mask] * self.width + x_pix[valid_mask]
            bev_img.reshape(-1, 3)[flat_indices] = [180, 180, 180]

    def _build_projection_matrix(self, cx=0.0, cy=0.0, yaw=0.0):
        """Build the affine transform from local coordinates to BEV pixels."""
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        res_inv = 1.0 / self.resolution
        return np.array([
            [-res_inv * sin_y, -res_inv * cos_y, res_inv * (self.max_y - cy)],
            [-res_inv * cos_y,  res_inv * sin_y, res_inv * (self.max_x - cx)]
        ], dtype=np.float32)

    def _render_spatial_grid_layers(self, bev_img):
        """Calculates sensor baseline frames and injects reference metric grids."""
        origin_px_x = int((self.max_y - 0.0) / self.resolution)
        origin_px_y = int((self.max_x - 0.0) / self.resolution)
        self._draw_metric_grid(bev_img, origin_px_x, origin_px_y)

    def _update_pipeline_fps(self):
        """Calculates streaming frame execution time using an exponential moving average."""
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            delta_time = current_time - self.last_frame_time
            if delta_time > 0.0:
                instantaneous_fps = 1.0 / delta_time
                self.current_fps = (self.alpha_fps * instantaneous_fps) + ((1.0 - self.alpha_fps) * self.current_fps)
        self.last_frame_time = current_time

    def _draw_diagnostic_dashboard(self, bev_img, render_cost_ms):
        """Renders alphanumeric diagnostic dashboard overlays for presentation tracking."""
        cv2.rectangle(bev_img, (5, 5), (145, 45), (15, 15, 18), -1)        
        cv2.putText(bev_img, f"PIPELINE: {self.current_fps:.1f} FPS", (12, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bev_img, f"BEV COST: {render_cost_ms:.1f} ms", (12, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    def _publish_bev_frame(self, bev_img, source_header, callback_start):
        """Encodes native NumPy canvas dimensions to standard ROS Image messages and publishes."""
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = source_header
        self.bev_pub.publish(bev_msg)
        
        total_cost_ms = (time.perf_counter() - callback_start) * 1000.0
        self.get_logger().info(
            f"Asynchronous BEV Render Complete. Total Pipeline Callback: {total_cost_ms:.2f}ms", 
            throttle_duration_sec=2.0
        )

    def _render_fused_detections(self, fused_msg, bev_img):
        """
        Renders 3D fused bounding boxes and dynamic tracking heading dots 
        onto the Bird's Eye View canvas using a single 2x3 affine matrix projection.
        """
        for det in fused_msg.detections:
            # Extract tracking coordinates from the 3D bounding box layer
            cx = float(det.bbox.center.position.x)  
            cy = float(det.bbox.center.position.y)
            sx = float(det.bbox.size.x)  # Object length along the x-axis.
            sy = float(det.bbox.size.y)  # Object width along the y-axis.

            # Enforce Region of Interest (ROI) filtering layer bounds
            if not (self.min_x <= cx <= self.max_x and self.min_y <= cy <= self.max_y):
                continue
                    
            distance_meters = np.hypot(cx, cy)
                
            # Unpack ROS orientation layout directly using standard order mapping
            q = det.bbox.center.orientation
            _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])

            # Define 2D perimeter corners inside the local coordinate frame
            local_corners = np.array([
                [ sx/2,  sy/2],  # Front-Left
                [ sx/2, -sy/2],  # Front-Right
                [-sx/2, -sy/2],  # Rear-Right
                [-sx/2,  sy/2]   # Rear-Left
            ], dtype=np.float32)

            # Construct a homogeneous local coordinate matrix [4x2] -> [4x3] (X_local, Y_local, 1)
            homogeneous_local = np.hstack([local_corners, np.ones((4, 1), dtype=np.float32)])

            # =========================================================================
            # 🌟 COMPOSITE 2x3 AFFINE TRANSFORMATION MATRIX (M) 🌟
            # Combines local yaw rotation, physical center translation, axis swapping, 
            # and asymmetric canvas resolution scaling in one single operation:
            # Pixels = M * [X_local, Y_local, 1]^T
            # =========================================================================

            # --- SPATIAL LANE-BASED HEADING CORRECTION LAYER ---
            # If the object is detected in the oncoming lane layout (cy > 0.5),
            # invert the rotation coefficients to enforce a 180-degree heading flip.
            if cy > 0.5:
                yaw += np.pi
            # ---------------------------------------------------
            M = self._build_projection_matrix(cx, cy, yaw)

            # Project local bounding box vertices directly into integer image pixels
            pix_corners = np.dot(M, homogeneous_local.T).T.astype(np.int32)
            # ========================================================================= 
                
            # Extract the detection class ID.
            cls_id = self._parse_class_id(det)                    

            # Use the class-specific style, or a green fallback for unknown classes.
            visual = CLASS_VISUALS.get(cls_id, {"name": f"ID:{cls_id}", "color": (0, 255, 0)})
                
            # Draw the projected bounding-box outline.
            cv2.polylines(bev_img, [pix_corners.reshape(-1, 1, 2)], isClosed=True, color=visual["color"], thickness=2)
                
            # Call heading draw pipeline
            f_mid_x, f_mid_y, center_pixel_x, center_pixel_y = self._draw_heading_vector(bev_img, cx, cy, sx, M, visual)

            # 🌟 FIX: Passed raw scalars directly without array subscription indexing
            self._log_heading_verification(distance_meters, q, yaw, cls_id, f_mid_x, f_mid_y, center_pixel_x, center_pixel_y)

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

    def _draw_heading_vector(self, bev_img, cx, cy, sx, M, visual):
        # Project structural front-mid profile reference dot using the identical M matrix
        front_mid_local = np.array([sx/2, 0.0, 1.0], dtype=np.float32)
        front_mid_px = np.dot(M, front_mid_local).astype(np.int32)
        
        # Render a solid tracking heading dot exactly over the object's bumper profile
        cv2.circle(bev_img, (front_mid_px[0], front_mid_px[1]), 5, visual["color"], -1)
        
        # Execute background diagnostics tracking verification
        center_pixel_x = int((self.max_y - cy) / self.resolution) # World y maps to pixel x.
        center_pixel_y = int((self.max_x - cx) / self.resolution) # World x maps to pixel y.
        
        return front_mid_px[0], front_mid_px[1], center_pixel_x, center_pixel_y

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
