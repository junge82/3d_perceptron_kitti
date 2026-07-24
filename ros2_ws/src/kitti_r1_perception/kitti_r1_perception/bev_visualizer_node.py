import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import transforms3d # Ensure you have this package installed (pip install transforms3d)

class BEVVisualizerNode(Node):
    def __init__(self):
        super().__init__('bev_visualizer_node')
        
        # BEV Parameters
        self.declare_parameter('range_x', 15.0) 
        self.declare_parameter('range_y', 15.0) 
        self.declare_parameter('resolution', 0.05) 
        
        self.range_x = self.get_parameter('range_x').value
        self.range_y = self.get_parameter('range_y').value
        self.resolution = self.get_parameter('resolution').value
        
        self.width = int(2 * self.range_y / self.resolution)
        self.height = int(2 * self.range_x / self.resolution)
        
        self.bridge = CvBridge()
        
        # Subscriptions
        self.lidar_sub = self.create_subscription(PointCloud2, '/kitti/velodyne/pointcloud', self.lidar_callback, 10)
        self.fused_sub = self.create_subscription(Detection3DArray, '/detection/fused_3d', self.fused_callback, 10)
        
        self.latest_fused = None
        
        # Publisher
        self.bev_pub = self.create_publisher(Image, '/visualization/bev_image', 10)
        
        self.get_logger().info(f'BEV Visualizer Node started. Image size: {self.width}x{self.height}')

    def fused_callback(self, msg):
        self.latest_fused = msg

    def lidar_callback(self, msg):
        # 1. Fast Convert PointCloud2 to numpy using built-in array operations
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        if points.size == 0:
            return

        # 2. Filter points by range
        mask = (points[:, 0] > -self.range_x) & (points[:, 0] < self.range_x) & \
               (points[:, 1] > -self.range_y) & (points[:, 1] < self.range_y)
        points = points[mask]

        # 3. Project to pixels
        x_pix = ((self.range_y - points[:, 1]) / self.resolution).astype(np.int32)
        y_pix = ((self.range_x - points[:, 0]) / self.resolution).astype(np.int32)

        # 4. Draw BEV Base
        bev_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw LiDAR points as grey pixels
        valid_mask = (x_pix >= 0) & (x_pix < self.width) & (y_pix >= 0) & (y_pix < self.height)
        bev_img[y_pix[valid_mask], x_pix[valid_mask]] = [200, 200, 200]

        # 5. Draw Fused Detections (Handling Rotations)
        if self.latest_fused:
            for det in self.latest_fused.detections:
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                sx = det.bbox.size.x  # Length (x-axis)
                sy = det.bbox.size.y  # Width (y-axis)
                
                # Extract orientation and calculate yaw angle
                q = det.bbox.center.orientation
                # transforms3d expects [w, x, y, z]
                _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
                
                # Define 4 local corners of the bounding box relative to its center
                # Order: Front-Left, Front-Right, Rear-Right, Rear-Left
                local_corners = np.array([
                    [ sx/2,  sy/2],
                    [ sx/2, -sy/2],
                    [-sx/2, -sy/2],
                    [-sx/2,  sy/2]
                ])
                
                # Create 2D rotation matrix from yaw
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                rot_mat = np.array([[cos_y, -sin_y], 
                                    [sin_y,  cos_y]])
                
                # Rotate and translate corners to world LiDAR positions
                world_corners = (rot_mat @ local_corners.T).T + np.array([cx, cy])
                
                # Project world corners to pixel coordinates
                pix_corners = np.zeros_like(world_corners, dtype=np.int32)
                pix_corners[:, 0] = ((self.range_y - world_corners[:, 1]) / self.resolution).astype(np.int32) # Column (px)
                pix_corners[:, 1] = ((self.range_x - world_corners[:, 0]) / self.resolution).astype(np.int32) # Row (py)
                
                # Draw the rotated bounding box
                cv2.polylines(bev_img, [pix_corners], isClosed=True, color=(0, 0, 255), thickness=2)
                
                # Draw heading direction vector indicator (Front-center line)
                front_mid_world = (rot_mat @ np.array([sx/2, 0])) + np.array([cx, cy])
                f_mid_px = int((self.range_y - front_mid_world[1]) / self.resolution)
                f_mid_py = int((self.range_x - front_mid_world[0]) / self.resolution)
                c_px = int((self.range_y - cy) / self.resolution)
                c_py = int((self.range_x - cx) / self.resolution)
                cv2.line(bev_img, (c_px, c_py), (f_mid_px, f_mid_py), (0, 255, 255), 2)

                # Define class mapping at the top of your callback or node
                class_mapping = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

                # Draw safe classification text label (Index 0)
                if det.results:
                    cls_id = int(det.results[0].hypothesis.class_id)
                    # Lookup class string, default to the ID string if unknown
                    cls_name = class_mapping.get(cls_id, f"ID:{cls_id}")
                    
                    cv2.putText(bev_img, cls_name, (pix_corners[0, 0], pix_corners[0, 1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 6. Publish BEV Image
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = msg.header
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
