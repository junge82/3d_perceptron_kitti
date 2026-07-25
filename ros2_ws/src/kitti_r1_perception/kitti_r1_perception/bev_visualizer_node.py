import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d # Ensure you have this package installed (pip install transforms3d)
# Import message filters for exact framework synchronization
import message_filters

class BEVVisualizerNode(Node):
    def __init__(self):
        super().__init__('bev_visualizer_node')
        
         # 1. EXACT ALIGNMENT WITH LIDAR_CLUSTER_NODE ROI PARAMETERS
        self.min_x = 1.5    # Forward minimum boundary
        self.max_x = 50.0   # Forward maximum boundary
        self.min_y = -25.0  # Right boundary
        self.max_y = 25.0   # Left boundary
        self.resolution = 0.10 # 10cm per pixel yields a clean 500x485 canvas layout

        # Compute exact pixel space boundaries based on the asymmetric box
        self.width = int((self.max_y - self.min_y) / self.resolution)   # Lateral width
        self.height = int((self.max_x - self.min_x) / self.resolution)  # Longitudinal height
        
        self.bridge = CvBridge()


        # QoS for incoming LiDAR and Fused 3D bounding boxes,
        # QoS for incoming synchronized topics (Must remain RELIABLE to match upstream)
        bev_sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,     # Must match upstream publishers!
            history=HistoryPolicy.KEEP_LAST,
            depth=5                                     # Matches frustum node's buffer allocation
            )
        
         # 1. Create message_filters Subscribers instead of traditional rclpy subscriptions
        self.lidar_sub = message_filters.Subscriber(
            self, PointCloud2, '/kitti/velodyne/pointcloud', qos_profile=bev_sub_qos
        )
        self.fused_sub = message_filters.Subscriber(
            self, Detection3DArray, '/detection/fused_3d', qos_profile=bev_sub_qos
        )

        # 2. Instantiate the Approximate Time Synchronizer
        # queue_size=10 gives the buffer room to align paired messages under load
        # slop=0.05 allows timestamp alignment variations up to 50ms (YOLO + Lidar cluster lag)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.fused_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        self.latest_fused = None

        # QoS for publishing your final visualization image to RViz2
        bev_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Images to RViz can be dropped to save bandwidth
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Publisher
        self.bev_pub = self.create_publisher(Image, '/visualization/bev_image', bev_pub_qos)
        
        self.get_logger().info(f'BEV Visualizer Node started. Image size: {self.width}x{self.height}')

    def synchronized_callback(self, lidar_msg, fused_msg):
        """Triggers exclusively when both messages are verified to share the same capture instance."""
        
        # 1. Convert PointCloud2 memory directly to standard NumPy format
        points = point_cloud2.read_points_numpy(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        if points.size == 0:
            return

        # 2. Filter raw points using the exact same ROI box from your cluster node
        mask = (points[:, 0] >= self.min_x) & (points[:, 0] <= self.max_x) & \
               (points[:, 1] >= self.min_y) & (points[:, 1] <= self.max_y)
        points = points[mask]

        # 3. Project world coordinates into aligned pixel columns and rows
        x_pix = ((self.max_y - points[:, 1]) / self.resolution).astype(np.int32)
        y_pix = ((self.max_x - points[:, 0]) / self.resolution).astype(np.int32)

        # 4. Generate base image canvas
        bev_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Paint within limits lidar measurements as gray pixels
        valid_mask = (x_pix >= 0) & (x_pix < self.width) & (y_pix >= 0) & (y_pix < self.height)
        bev_img[y_pix[valid_mask], x_pix[valid_mask]] = [200, 200, 200]

        # --- NEW: DRAW CONCENTRIC DISTANCE RINGS & METERS OVERLAYS ---
        # Real-world sensor origin (0,0) projected into our asymmetric pixel coordinate frame
        origin_px_x = int((self.max_y - 0.0) / self.resolution)
        origin_px_y = int((self.max_x - 0.0) / self.resolution)
        
        # Target distance thresholds in meters
        ring_distances_meters = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for dist in ring_distances_meters:
            # Convert real-world meter radius directly to pixel radius
            radius_px = int(dist / self.resolution)
            
            # Draw semi-transparent dark grey reference circles
            cv2.circle(bev_img, (origin_px_x, origin_px_y), radius_px, (70, 70, 70), thickness=1, lineType=cv2.LINE_AA)
            
            # Calculate text drawing anchor point (placed along the center forward centerline)
            text_y_px = int((self.max_x - dist) / self.resolution)
            
            # Only draw the text label if it falls inside the visible image canvas boundary
            if 0 <= text_y_px < self.height:
                # Add text backing outline shadow for high visibility against dense point clouds
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                # Clear white text overlay
                cv2.putText(bev_img, f"{int(dist)}m", (origin_px_x + 5, text_y_px + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)


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
        
        for det in fused_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size.x  # Length (x-axis)
            sy = det.bbox.size.y  # Width (y-axis)
            
            # Skip drawing immediately if the object center is outside the operational cluster ROI
            if not (self.min_x <= cx <= self.max_x and self.min_y <= cy <= self.max_y):
                continue
                
            distance_meters = np.hypot(cx, cy)
            
            # Unpack quaternion layout to calculate object yaw heading rotation angle
            q = det.bbox.center.orientation
            _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
            
            # Generate the bounding box perimeter corners relative to its local center coordinate
            local_corners = np.array([
                [ sx/2,  sy/2], [ sx/2, -sy/2],
                [-sx/2, -sy/2], [-sx/2,  sy/2]
            ])
            
            # Build 2D rotation matrix using computed yaw
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[cos_y, -sin_y], 
                                [sin_y,  cos_y]])
            
            # Perform matrix transform to world coordinates
            world_corners = (rot_mat @ local_corners.T).T + np.array([cx, cy])
            
            # Project transformed corners into the aligned asymmetric pixel space
            pix_corners = np.zeros_like(world_corners, dtype=np.int32)
            pix_corners[:, 0] = ((self.max_y - world_corners[:, 1]) / self.resolution).astype(np.int32) 
            pix_corners[:, 1] = ((self.max_x - world_corners[:, 0]) / self.resolution).astype(np.int32) 
            
            # Safe Portable Class Extraction mapping ROS 2 vision_msg API changes safely
            cls_id = -1
            if det.results and len(det.results) > 0:
                first_result = det.results[0]  # Hardcoded target index 0 fixes list parsing bugs
                
                if hasattr(first_result, 'hypothesis'):
                    cls_id = int(first_result.hypothesis.class_id)
                else:
                    cls_id = int(first_result.class_id)

            visual = CLASS_VISUALS.get(cls_id, {"name": f"ID:{cls_id}", "color": (0, 255, 0)})
            
            # Draw rotated boundary box using class-specific color
            cv2.polylines(bev_img, [pix_corners], isClosed=True, color=visual["color"], thickness=2)
            
            # Draw forward heading direction vector indicator line
            front_mid_world = (rot_mat @ np.array([sx/2, 0])) + np.array([cx, cy])
            f_mid_px = int((self.max_y - front_mid_world[1]) / self.resolution)
            f_mid_py = int((self.max_x - front_mid_world[0]) / self.resolution)
            c_px = int((self.max_y - cy) / self.resolution)
            c_py = int((self.max_x - cx) / self.resolution)
            cv2.line(bev_img, (c_px, c_py), (f_mid_px, f_mid_py), visual["color"], 2)

            # Draw text label layout (Cleaned and optimized Single Block)
            display_text = f"{visual['name']} [{distance_meters:.1f}m]"
            tx = int(pix_corners[0, 0])
            ty = int(pix_corners[0, 1] - 5)
            cv2.putText(bev_img, display_text, (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, visual["color"], 1)

        # 6. Ship image message using Lidar header metadata
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = lidar_msg.header
        self.bev_pub.publish(bev_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BEVVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
