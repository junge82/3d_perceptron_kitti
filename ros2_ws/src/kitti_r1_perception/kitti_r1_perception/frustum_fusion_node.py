import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from vision_msgs.msg import Detection3DArray, Detection2DArray, ObjectHypothesisWithPose
import numpy as np
import message_filters
import torch 
import time  # Built-in high-precision CPU timer

class FrustumFusionNode(Node):
    def __init__(self):
        super().__init__('frustum_fusion_node')

        # ⚡ CRITICAL FIX: Declare tracking variables first before any callbacks spawn
        self.latest_camera_msg = None

        # ⚡ CRITICAL FIX: Force unbuffered stdout line matching stream sync
        import sys
        sys.stdout.reconfigure(line_buffering=True)

        # Parameters
        self.declare_parameter('match_radius', 50.0) 
        self.match_radius = self.get_parameter('match_radius').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # CUDA Events for precise GPU benchmarking
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

        # ⚡ FIX 1: Change to RELIABLE to establish an active link with your up-stream nodes
        optimized_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5 # Healthy depth allowance for synchronizer pairing buffers
        )

        # ⚡ FIX 2: Change to ReentrantCallbackGroup so both subscription channels 
        # can safely fill the synchronizer queue concurrently without locking each other out
        self.callback_group = ReentrantCallbackGroup()

        # Direct Subscriptions (No slow filtering middleware locks)
        self.camera_sub = self.create_subscription(
            Detection2DArray, '/detection/camera_2d',
            self.camera_callback, optimized_qos, callback_group=self.callback_group
        )

        self.lidar_sub = self.create_subscription(
            Detection3DArray, '/detection/lidar_clusters',
            self.lidar_callback, optimized_qos, callback_group=self.callback_group
        )

        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 2)
        self.get_logger().info('Frustum Fusion Node with CUDA-aware performance tracking with E2E Age Tracking started.')

    def camera_callback(self, msg):
        """Thread 1: Instantly caches the incoming 2D boxes in RAM."""
        self.latest_camera_msg = msg

    def lidar_callback(self, lidar_msg):
        # Start callback CPU timer
        callback_start_time = time.perf_counter()
        """Thread 2: Triggered by Lidar clusters, pairs immediately with the latest camera frame."""
        camera_msg = self.latest_camera_msg

        # Escape safely if the camera hasn't published anything yet
        # ⚡ OPTIMIZATION: Check if YOLO is still initializing its memory layout on startup
        if camera_msg is None:
            self.get_logger().info(
                "Fusion Layer Status: [WARMUP] Waiting for YOLOv8 weights initialization to complete...",
                throttle_duration_sec=1.5  # 💡 Throttles log output to prevent console clutter
            )
            return

        self.match_radius = self.get_parameter('match_radius').value
        num_3d = len(lidar_msg.detections)
        num_2d = len(camera_msg.detections)

        # if num_3d == 0 or num_2d == 0:
        #     return

        if self.is_cuda:
            self.start_event.record()
        else:
            gpu_start_cpu_fallback = time.perf_counter()

        # Vectorized 3D Lidar Extraction
        lidar_pts_np = np.zeros((num_3d, 3), dtype=np.float32)
        for idx in range(num_3d):
            pos = lidar_msg.detections[idx].bbox.center.position
            lidar_pts_np[idx, 0] = pos.x
            lidar_pts_np[idx, 1] = pos.y
            lidar_pts_np[idx, 2] = pos.z
        pts_3d_lidar = torch.as_tensor(lidar_pts_np, device=self.device)

        # 2. Transform 3D Lidar Points to Camera Frame
        pts_3d_cam = torch.addmm(self.t_tensor, pts_3d_lidar, self.R_tensor.t())

        # 3. Filter points behind the camera
        valid_mask = pts_3d_cam[:, 2] > 0.1
        if not torch.any(valid_mask):
            self.get_logger().info("Sync Match: No valid 3D points in camera frame, skipping frame.")
            return

        # ⚡ FIX 1: Extract the index element list array out of the tuple slice
        valid_indices = torch.where(valid_mask)[0]
        pts_3d_cam_filtered = pts_3d_cam[valid_indices]

        # 4. Project 3D Camera Points into 2D Pixel Coordinates
        proj_pts = torch.addmm(torch.zeros(3, device=self.device), pts_3d_cam_filtered, self.K_tensor.t())
        inv_z = 1.0 / proj_pts[:, 2:3]
        uv_gpu = proj_pts[:, :2] * inv_z

        # ⚡ FIX 4: True Vectorized Zero-Copy 2D Center Box Extraction
        camera_pts_np = np.zeros((num_2d, 2), dtype=np.float32)
        for idx in range(num_2d):
            center_pos = camera_msg.detections[idx].bbox.center.position
            camera_pts_np[idx, 0] = center_pos.x
            camera_pts_np[idx, 1] = center_pos.y
        pts_2d_tensor = torch.as_tensor(camera_pts_np, device=self.device)

        # 6. Compute Parallel Distance Matrix via GPU Broadcasting
        diff = uv_gpu.unsqueeze(1) - pts_2d_tensor.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=2) 
        min_dists, best_2d_indices = torch.min(dist_matrix, dim=1)

        # Bring tracking configurations back to host space
        min_dists_cpu = min_dists.cpu().numpy()
        best_2d_indices_cpu = best_2d_indices.cpu().numpy()
        
        # ⚡ FIX 2: Force the unpacked index array to stay flat (Prevents scalar 0-d drops)
        valid_indices_cpu = valid_indices.cpu().numpy().flatten()

        # ----------------------------------------------------
        # END METRIC: GPU Calculations Boundary
        # ----------------------------------------------------
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()  
            gpu_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            gpu_time_ms = (time.perf_counter() - gpu_start_cpu_fallback) * 1000.0

        # 7. Construct final fused message
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
            # Print the success trace to your console output mapping
            self.get_logger().info(
                f"\n--- ⚡ GPU FRUSTUM FUSION TARGET TRACE ⚡ ---\n"
                f"Incoming 3D Clusters : {num_3d}\n"
                f"Incoming 2D Boxes    : {num_2d}\n"
                f"Successfully Fused   : {len(fused_detections.detections)}\n"
                f"GPU Processing Time  : {gpu_time_ms:.2f} ms\n"
                f"-----------------------------------------\n"
            )
        # ----------------------------------------------------
        # LATENCY & AGE CALCULATIONS
        # ----------------------------------------------------
        # 1. Total time inside this specific node callback
        callback_time_ms = (time.perf_counter() - callback_start_time) * 1000.0

        # 2. Extract accurate ROS system clock time at the moment of publishing
        now = self.get_clock().now()

        # 3. Convert message header timestamps to rclpy Time objects
        lidar_stamp = rclpy.time.Time.from_msg(lidar_msg.header.stamp, 
            clock_type=rclpy.time.ClockType.ROS_TIME)
        camera_stamp = rclpy.time.Time.from_msg(camera_msg.header.stamp, 
            clock_type=rclpy.time.ClockType.ROS_TIME)

        # 4. Calculate total E2E processing age (Sensor Capture -> Node Finished)
        lidar_e2e_age_ms = (now - lidar_stamp).nanoseconds / 1e6
        camera_e2e_age_ms = (now - camera_stamp).nanoseconds / 1e6

        # 5. Measure the synchronization shift between the two sensor packets
        sync_slop_ms = abs((lidar_stamp - camera_stamp).nanoseconds / 1e6)

        # Print all metrics clearly categorized
        self.get_logger().info(
            f"\n--- Fusion PERFORMANCE PROFILE ---\n"
            f"Active Objects     : 3D: {num_3d} | 2D: {num_2d}\n"
            f"Node Callback Cost : GPU Math: {gpu_time_ms:.2f}ms | Total Callback: {callback_time_ms:.2f}ms\n"
            f"Synchronizer Slop  : {sync_slop_ms:.2f}ms\n"
            f"End-to-End Stack Age: LiDAR E2E: {lidar_e2e_age_ms:.2f}ms | Camera E2E: {camera_e2e_age_ms:.2f}ms"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FrustumFusionNode()

    # 3. Spin up the Multi-Threaded environment
    executor = MultiThreadedExecutor(num_threads=2)  # Adjust thread count based on system capabilities
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        
        # FIXED: Enforce clear context protection guard for ROS 2 Jazzy
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
