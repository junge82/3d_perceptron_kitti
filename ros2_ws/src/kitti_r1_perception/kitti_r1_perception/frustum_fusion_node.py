import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import message_filters
import torch # Added for GPU acceleration

class FrustumFusionNode(Node):
    def __init__(self):
        super().__init__('frustum_fusion_node')
        
        # Parameters
        self.declare_parameter('lidar_frame', 'velodyne')
        self.declare_parameter('camera_frame', 'zed2_left_camera_optical_frame')
        self.declare_parameter('match_radius', 50.0) # Search radius in pixels
        
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.match_radius = self.get_parameter('match_radius').value

        # Select GPU device if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics matrix K pushed to GPU
        K_kitti_np = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.K_tensor = torch.from_numpy(K_kitti_np).to(self.device)

        # Synchronized Subscriptions
        self.lidar_sub = message_filters.Subscriber(self, Detection3DArray, '/detection/lidar_clusters')
        self.camera_sub = message_filters.Subscriber(self, Detection2DArray, '/detection/camera_2d')
        
        # Use approximate sync since timestamps might not be identical in bag files
        self.sync = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.camera_sub], 10, 0.1)
        self.sync.registerCallback(self.fused_callback)

        # Publisher
        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 10)
        
        self.get_logger().info('Frustum Fusion Node started.')

    def fused_callback(self, lidar_msg, camera_msg):
        # Live-update parameters from the ROS 2 server
        self.match_radius = self.get_parameter('match_radius').value

        if self.K_tensor is None:
            self.get_logger().warn('No camera info received yet.')
            return

        # Look up transform from Lidar to Camera
        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                lidar_msg.header.frame_id,
                lidar_msg.header.stamp,
                rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().error(f'Could not lookup transform: {e}')
            return

         # 1. Transform all 3D points on CPU (tf2 requires this)
        valid_3d_points = []
        valid_3d_indices = []

        for idx, det_3d in enumerate(lidar_msg.detections):
            point_lidar = PointStamped()
            point_lidar.header = lidar_msg.header
            point_lidar.point = det_3d.bbox.center.position
            
            point_cam = tf2_geometry_msgs.do_transform_point(point_lidar, transform)
            
            # Filter out points behind camera immediately
            if point_cam.point.z > 0:
                valid_3d_points.append([point_cam.point.x, point_cam.point.y, point_cam.point.z])
                valid_3d_indices.append(idx)

        if not valid_3d_points:
            return

        # 2. Extract 2D camera detection centers
        points_2d = [[det.bbox.center.position.x, det.bbox.center.position.y] for det in camera_msg.detections]

        # 3. Push data arrays to GPU Tensors
        pts_3d_tensor = torch.tensor(valid_3d_points, dtype=torch.float32, device=self.device) # Shape: (N, 3)
        pts_2d_tensor = torch.tensor(points_2d, dtype=torch.float32, device=self.device)      # Shape: (M, 2)

        # 4. GPU Project: Batch multiplication and division
        # Transpose pts_3d to (3, N) for matrix multiplication: K @ P
        proj_pts = torch.matmul(self.K_tensor, pts_3d_tensor.t()) # Shape: (3, N)
        u_gpu = proj_pts[0] / proj_pts[2]
        v_gpu = proj_pts[1] / proj_pts[2]
        uv_gpu = torch.stack([u_gpu, v_gpu], dim=1) # Shape: (N, 2)

        # 5. GPU Distance Matrix: Compute all distances in one parallel operation
        # (N, 1, 2) - (1, M, 2) utilizing broadcasting -> Shape: (N, M, 2)
        diff = uv_gpu.unsqueeze(1) - pts_2d_tensor.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=2) # Shape: (N, M)

        # Find closest 2D box for every 3D cluster
        min_dists, best_2d_indices = torch.min(dist_matrix, dim=1)

        # Move small validation results back to CPU
        min_dists = min_dists.cpu().numpy()
        best_2d_indices = best_2d_indices.cpu().numpy()

        # 6. Build final fused message
        fused_detections = Detection3DArray()
        fused_detections.header = lidar_msg.header

        for i, lidar_idx in enumerate(valid_3d_indices):
            if min_dists[i] < self.match_radius:
                det_3d = lidar_msg.detections[lidar_idx]
                best_match_2d = camera_msg.detections[best_2d_indices[i]]

                hypothesis = det_3d.results[0] if det_3d.results else ObjectHypothesisWithPose() 
                
                for res_2d in best_match_2d.results:
                    hypothesis.hypothesis.class_id = res_2d.hypothesis.class_id
                    hypothesis.hypothesis.score = res_2d.hypothesis.score
                    det_3d.results.append(hypothesis)

                fused_detections.detections.append(det_3d)

        self.fused_pub.publish(fused_detections)

def main(args=None):
    rclpy.init(args=args)
    node = FrustumFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
