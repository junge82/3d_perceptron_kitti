import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection2DArray, ObjectHypothesisWithPose
import numpy as np
import message_filters
import torch 
import time  # Built-in high-precision CPU timer

class FrustumFusionNode(Node):
    def __init__(self):
        super().__init__('frustum_fusion_node')
        
        # Parameters
        self.declare_parameter('match_radius', 50.0) 
        self.match_radius = self.get_parameter('match_radius').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Create CUDA Events for precise GPU benchmarking (ignored seamlessly if using CPU)
        self.is_cuda = self.device.type == 'cuda'
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        # 1. Camera Intrinsics Matrix (K)
        K_kitti_np = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.K_tensor = torch.from_numpy(K_kitti_np).to(self.device)

        # 2. Extrinsic Transformation Matrix [R | t]
        T_lidar_to_cam_np = np.array([
            [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
            [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
        ], dtype=np.float32)
        
        self.R_tensor = torch.from_numpy(T_lidar_to_cam_np[:3, :3]).to(self.device)
        self.t_tensor = torch.from_numpy(T_lidar_to_cam_np[:3, 3]).to(self.device)

        # Synchronized Subscriptions
        self.lidar_sub = message_filters.Subscriber(self, Detection3DArray, '/detection/lidar_clusters')
        self.camera_sub = message_filters.Subscriber(self, Detection2DArray, '/detection/camera_2d')
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.camera_sub], queue_size=2, slop=0.05)
        self.sync.registerCallback(self.fused_callback)

        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 2)
        self.get_logger().info('Frustum Fusion Node with CUDA-aware performance tracking started.')

    def fused_callback(self, lidar_msg, camera_msg):
        # Start overall callback CPU timer
        callback_start_time = time.perf_counter()

        self.match_radius = self.get_parameter('match_radius').value
        num_3d = len(lidar_msg.detections)
        num_2d = len(camera_msg.detections)

        if num_3d == 0 or num_2d == 0:
            return

        # ----------------------------------------------------
        # START METRIC: GPU Calculations Boundary
        # ----------------------------------------------------
        if self.is_cuda:
            self.start_event.record()
        else:
            gpu_start_cpu_fallback = time.perf_counter()

        # 1. 3D Lidar Extraction
        pts_3d_flat = [val for d in lidar_msg.detections for val in (d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z)]
        lidar_pts_np = np.fromiter(pts_3d_flat, dtype=np.float32, count=num_3d * 3).reshape(num_3d, 3)
        pts_3d_lidar = torch.as_tensor(lidar_pts_np, device=self.device)

        # 2. Transform 3D Lidar Points to Camera Frame
        pts_3d_cam = torch.addmm(self.t_tensor, pts_3d_lidar, self.R_tensor.t())

        # 3. Filter points behind the camera
        valid_mask = pts_3d_cam[:, 2] > 0.1
        if not torch.any(valid_mask):
            return
        valid_indices = torch.where(valid_mask)
        pts_3d_cam_filtered = pts_3d_cam[valid_indices]

        # 4. Project 3D Camera Points into 2D Pixel Coordinates
        proj_pts = torch.addmm(torch.zeros(3, device=self.device), pts_3d_cam_filtered, self.K_tensor.t())
        inv_z = 1.0 / proj_pts[:, 2:3]
        uv_gpu = proj_pts[:, :2] * inv_z

        # 5. 2D Camera Extraction
        pts_2d_flat = [val for det in camera_msg.detections for val in (det.bbox.center.position.x, det.bbox.center.position.y)]
        camera_pts_np = np.fromiter(pts_2d_flat, dtype=np.float32, count=num_2d * 2).reshape(num_2d, 2)
        pts_2d_tensor = torch.as_tensor(camera_pts_np, device=self.device)

        # 6. Compute Parallel Distance Matrix
        diff = uv_gpu.unsqueeze(1) - pts_2d_tensor.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=2) 
        min_dists, best_2d_indices = torch.min(dist_matrix, dim=1)

        # Bring tracking configurations back to host space (FIXED: valid_indices[0] used for tuple extraction)
        min_dists_cpu = min_dists.cpu().numpy()
        best_2d_indices_cpu = best_2d_indices.cpu().numpy()
        valid_indices_cpu = valid_indices[0].cpu().numpy()

        # ----------------------------------------------------
        # END METRIC: GPU Calculations Boundary
        # ----------------------------------------------------
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()  
            gpu_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            gpu_time_ms = (time.perf_counter() - gpu_start_cpu_fallback) * 1000.0

        # 7. Construct final fused message (FIXED API assignments)
        fused_detections = Detection3DArray()
        fused_detections.header = lidar_msg.header

        for i, lidar_idx in enumerate(valid_indices_cpu):
            if min_dists_cpu[i] < self.match_radius:
                det_3d = lidar_msg.detections[lidar_idx]
                best_match_2d = camera_msg.detections[best_2d_indices_cpu[i]]

                # Clear previous results to prevent stacking duplicates
                det_3d.results = []

                for res_2d in best_match_2d.results:
                    inst_hypothesis = ObjectHypothesisWithPose()
                    
                    # Safely handle both Humble and Jazzy vision_msgs variants
                    if hasattr(res_2d, 'hypothesis'):
                        inst_hypothesis.hypothesis.class_id = res_2d.hypothesis.class_id
                        inst_hypothesis.hypothesis.score = res_2d.hypothesis.score
                    else:
                        # Direct property fallback
                        inst_hypothesis.class_id = res_2d.class_id
                        inst_hypothesis.score = res_2d.score
                        
                    det_3d.results.append(inst_hypothesis)

                fused_detections.detections.append(det_3d)

        if len(fused_detections.detections) > 0:
            self.fused_pub.publish(fused_detections)

        # End overall callback CPU timer
        callback_time_ms = (time.perf_counter() - callback_start_time) * 1000.0

        # Report latencies
        self.get_logger().info(
            f"[PERF] Objects(3D/2D): {num_3d}/{num_2d} | "
            f"Pure GPU Math: {gpu_time_ms:.2f}ms | "
            f"Total Callback: {callback_time_ms:.2f}ms"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FrustumFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
