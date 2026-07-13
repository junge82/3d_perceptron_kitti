import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection2DArray, ObjectHypothesisWithPose
import numpy as np
import message_filters
import torch 

class FrustumFusionNode(Node):
    def __init__(self):
        super().__init__('frustum_fusion_node')
        
        # Parameters
        self.declare_parameter('match_radius', 50.0) # Search radius in pixels
        self.match_radius = self.get_parameter('match_radius').value

        # Select GPU device if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # 1. Camera Intrinsics Matrix (K)
        K_kitti_np = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.K_tensor = torch.from_numpy(K_kitti_np).to(self.device)

        # 2. Extrinsic Transformation Matrix [R | t] (Lidar -> Camera Frame)
        # Parsed exactly from your KITTI calib_velo_to_cam parameters
        T_lidar_to_cam_np = np.array([
            [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
            [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
        ], dtype=np.float32)
        
        # Extract R (3x3) and t (3x1) for faster GPU math
        self.R_tensor = torch.from_numpy(T_lidar_to_cam_np[:3, :3]).to(self.device)
        self.t_tensor = torch.from_numpy(T_lidar_to_cam_np[:3, 3]).to(self.device)

        # Synchronized Subscriptions
        self.lidar_sub = message_filters.Subscriber(self, Detection3DArray, '/detection/lidar_clusters')
        self.camera_sub = message_filters.Subscriber(self, Detection2DArray, '/detection/camera_2d')
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.camera_sub], 10, 0.1)
        self.sync.registerCallback(self.fused_callback)

        # Publisher
        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 10)
        self.get_logger().info('Frustum Fusion Node (Pure GPU, TF-free) started.')

    def fused_callback(self, lidar_msg, camera_msg):
        self.match_radius = self.get_parameter('match_radius').value

        if not lidar_msg.detections or not camera_msg.detections:
            return

        # 1. Extract 3D Lidar Centers directly into a GPU tensor
        lidar_pts = [[d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z] 
                     for d in lidar_msg.detections]
        pts_3d_lidar = torch.tensor(lidar_pts, dtype=torch.float32, device=self.device) # Shape: (N, 3)

        # 2. Transform 3D Lidar Points to Camera Frame entirely on GPU: P_cam = R @ P_lidar + t
        pts_3d_cam = torch.matmul(pts_3d_lidar, self.R_tensor.t()) + self.t_tensor # Shape: (N, 3)

        # 3. Filter points behind the camera using a mask
        valid_mask = pts_3d_cam[:, 2] > 0.1 # Keep points with positive depth
        if not torch.any(valid_mask):
            return

        # Filter points and save indices to match back to the ROS message later
        pts_3d_cam_filtered = pts_3d_cam[valid_mask] # Shape: (K, 3)
        valid_indices = torch.where(valid_mask)[0].cpu().numpy()

        # 4. Project 3D Camera Points into 2D Pixel Coordinates
        proj_pts = torch.matmul(pts_3d_cam_filtered, self.K_tensor.t()) # Shape: (K, 3)
        u_gpu = proj_pts[:, 0] / proj_pts[:, 2]
        v_gpu = proj_pts[:, 1] / proj_pts[:, 2]
        uv_gpu = torch.stack([u_gpu, v_gpu], dim=1) # Shape: (K, 2)

        # 5. Extract 2D Camera Bounding Box Centers
        points_2d = [[det.bbox.center.position.x, det.bbox.center.position.y] for det in camera_msg.detections]
        pts_2d_tensor = torch.tensor(points_2d, dtype=torch.float32, device=self.device) # Shape: (M, 2)

        # 6. Compute parallel Distance Matrix on GPU
        diff = uv_gpu.unsqueeze(1) - pts_2d_tensor.unsqueeze(0) # Shape: (K, M, 2)
        dist_matrix = torch.norm(diff, dim=2) # Shape: (K, M)

        # Find closest 2D box for every remaining 3D cluster
        min_dists, best_2d_indices = torch.min(dist_matrix, dim=1)

        # Bring mapping data back to CPU
        min_dists = min_dists.cpu().numpy()
        best_2d_indices = best_2d_indices.cpu().numpy()

        # 7. Build final fused message
        fused_detections = Detection3DArray()
        fused_detections.header = lidar_msg.header

        for i, lidar_idx in enumerate(valid_indices):
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
