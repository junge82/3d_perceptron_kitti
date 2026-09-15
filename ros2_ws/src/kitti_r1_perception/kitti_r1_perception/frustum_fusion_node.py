import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
        self.declare_parameter('match_radius', 50.0) # Matches your projection scaling limits
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

        # ⚡ FIX 2: Use a Mutually Exclusive Callback Group to prevent deadlocks in multi-threaded execution
        # ⚡ ISOLATE THREAD QUEUE TO PREVENT RVIZ2 STALLING
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # Publisher for fused 3D detections
        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 2)

        # # TRUE SYNCHRONIZATION: Use message_filters to pair 2D and 3D detections based on timestamps
        self.camera_filter = message_filters.Subscriber(self,
            Detection2DArray, 
            '/detection/camera_2d',
            qos_profile=optimized_qos,
            callback_group=self.callback_group)

        self.lidar_filter = message_filters.Subscriber(self,
            Detection3DArray,
            '/detection/lidar_clusters',
            qos_profile=optimized_qos,
            callback_group=self.callback_group)

        # Use ApproximateTimeSynchronizer to allow for slight timestamp mismatches
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_filter, self.lidar_filter],
            queue_size=3, # Small queue stops backup leakage completely
            slop=0.03,  # Forces strict 30ms realt-time frame pairing, but allows for slight jitter
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Vectorized CUDA Frustum Fusion Engine Activated Successfully.")

    def sync_callback(self, camera_msg, lidar_msg):
        # Start callback CPU timer
        callback_start_time = time.perf_counter()        

        # Escape safely if the camera hasn't published anything yet        
        if camera_msg is None:
            self.get_logger().info(
                "Fusion Layer Status: [WARMUP] Waiting for camera msgs...",
                throttle_duration_sec=1.5  # 💡 Throttles log output to prevent console clutter
            )
            return

        self.match_radius = self.get_parameter('match_radius').value
        num_3d = len(lidar_msg.detections)
        num_2d = len(camera_msg.detections)

        if num_3d == 0 or num_2d == 0:
            return

        if self.is_cuda:
            self.start_event.record()
        else:
            gpu_start_cpu_fallback = time.perf_counter()

        # ⚡ ADAPTIVE UNPACKER: Adapts to your exact ONNX Camera Node message structure
        try:
            # 1. Extract 3D LiDAR Points
            lidar_pts_np = np.array([
                [float(d.bbox.center.position.x), float(d.bbox.center.position.y), float(d.bbox.center.position.z)]
                for d in lidar_msg.detections
            ], dtype=np.float32)
            
            # 2. Extract 2D Camera Points adaptively
            camera_pts = []
            for c in camera_msg.detections:
                # Check if the center object uses .position layer (like some 3D messages or custom shapes)
                if hasattr(c.bbox.center, 'position'):
                    cx = float(c.bbox.center.position.x)
                    cy = float(c.bbox.center.position.y)
                elif hasattr(c.bbox.center, 'x'):
                    cx = float(c.bbox.center.x)
                    cy = float(c.bbox.center.y)
                else:
                    # Deep inspection if custom attributes are wrapped inside
                    self.get_logger().error(f"Unknown center schema attributes: {dir(c.bbox.center)}")
                    return
                camera_pts.append([cx, cy])
                
            camera_pts_np = np.array(camera_pts, dtype=np.float32)
            
        except Exception as e:
            self.get_logger().error(f"🚨 Structure parsing exception: {str(e)}")
            return

        pts_3d_lidar = torch.as_tensor(lidar_pts_np, device=self.device)
        pts_2d_tensor = torch.as_tensor(camera_pts_np, device=self.device)

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
        # ⚡ 1. FIXED ADDMM: Pre-allocate bias vector matching the exact [N, 3] shape dynamically
        # Using a flat zeros vector initialized to the current valid point shape prevents shape mismatches.
        num_valid_pts = pts_3d_cam_filtered.shape[0]
        bias_tensor = torch.zeros((num_valid_pts, 3), dtype=torch.float32, device=self.device)
        
        try:
            proj_pts = torch.addmm(bias_tensor, pts_3d_cam_filtered, self.K_tensor.t())
            
            # ⚡ 2. PROTECTION: Clamp Z values to prevent Division-by-Zero / NaN spikes
            # Enforcing a strict minimum ceiling of 1e-4 blocks any reciprocal float overflows.
            z_clamped = torch.clamp(proj_pts[:, 2:3], min=1e-4)
            inv_z = 1.0 / z_clamped
            
            # Project onto the 2D image coordinates plane
            uv_gpu = proj_pts[:, :2] * inv_z

            # ⚡ NEW: VECTORIZED CAMERA FIELD-OF-VIEW CLIPPING MASK
            # Creates a single boolean mask on the GPU to drop anything outside the image borders
            in_fov_mask = (uv_gpu[:, 0] >= 0.0) & (uv_gpu[:, 0] <= 1248.0) & \
                      (uv_gpu[:, 1] >= 0.0) & (uv_gpu[:, 1] <= 384.0)

            # Apply the mask instantly to slice your tracking points down to visible elements
            valid_indices = torch.where(in_fov_mask)[0]
            uv_gpu_clipped = uv_gpu[valid_indices]
            
        except RuntimeError as e:
            self.get_logger().error(f"🚨 Matrix Multiplication Shape Mismatch inside Fusion Layer: {str(e)}")
            return
        except Exception as e:
            self.get_logger().error(f"🚨 Projection Execution Exception: {str(e)}")
            return

        # ⚡ QUICK CODES FOR COORDINATES PROJECTION VERIFICATION
        # Extract the very first projected point coordinates from the GPU matrix view
        if uv_gpu_clipped.shape[0] > 0:
            first_u = float(uv_gpu_clipped[0, 0].item())
            first_v = float(uv_gpu_clipped[0, 1].item())
            
            # Print the data to make sure it tracks inside your target camera bounds
            self.get_logger().info(
                f"[PROJECTION ALIGNMENT CHECK] Projected 2D Pixel: (u={first_u:.1f}, v={first_v:.1f}) | "
                f"Valid 1248x384 Range: {0.0 <= first_u <= 1248.0 and 0.0 <= first_v <= 384.0}",
                throttle_duration_sec=2.0
            )
        

        # 6. Compute Parallel Distance Matrix via GPU Broadcasting
        diff = uv_gpu_clipped.unsqueeze(1) - pts_2d_tensor.unsqueeze(0)
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

                # 1. Unpack real-world physical boundaries from the LiDAR box
                sx = det_3d.bbox.size.x  # Longitudinal size (Length)
                sy = det_3d.bbox.size.y  # Lateral size (Width)
                sz = det_3d.bbox.size.z  # Vertical size (Height)
                
                # 2. Compute advanced geometric ratios [1]
                # A car footprint area is large; a bike is a narrow line strip [1]
                horizontal_footprint_area = sx * sy
                aspect_ratio = max(sx, sy) / max(min(sx, sy), 0.001) # Length-to-width ratio
                
                # Clear previous results to prevent stacking duplicates
                det_3d.results = []

                for res_2d in best_match_2d.results:
                    inst_hypothesis = ObjectHypothesisWithPose()
                    
                    # Unpack raw class assignment string primitive from camera tracking
                    if hasattr(res_2d, 'hypothesis'):
                        camera_class = int(res_2d.hypothesis.class_id)
                        score = res_2d.hypothesis.score
                    else:
                        camera_class = int(res_2d.class_id)
                        score = res_2d.score

                    self.box_class_det(sx, sy, horizontal_footprint_area, aspect_ratio, inst_hypothesis, camera_class, score)
                        
                    det_3d.results.append(inst_hypothesis)

                # Append the verified detection to the publishing queue
                fused_detections.detections.append(det_3d)

        # Publish results if any objects were successfully projected and tracked
        if len(fused_detections.detections) > 0:
            self.fused_pub.publish(fused_detections)
            self.get_logger().info(
                f"\n--- ⚡ GPU FRUSTUM FUSION TARGET TRACE ⚡ ---\n"
                f"Incoming 3D Clusters : {num_3d}\n"
                f"Incoming 2D Boxes    : {num_2d}\n"
                f"Successfully Fused   : {len(fused_detections.detections)}\n"
                f"GPU Processing Time  : {gpu_time_ms:.2f} ms\n"
                f"-----------------------------------------\n"
            )
        # # ----------------------------------------------------
        # # LATENCY & AGE CALCULATIONS
        # # ----------------------------------------------------
        # # 1. Total time inside this specific node callback
        # callback_time_ms = (time.perf_counter() - callback_start_time) * 1000.0

        # # 2. Extract accurate ROS system clock time at the moment of publishing
        # now = self.get_clock().now()

        # # 3. Convert message header timestamps to rclpy Time objects
        # lidar_stamp = rclpy.time.Time.from_msg(lidar_msg.header.stamp, 
        #     clock_type=rclpy.time.ClockType.ROS_TIME)
        # camera_stamp = rclpy.time.Time.from_msg(camera_msg.header.stamp, 
        #     clock_type=rclpy.time.ClockType.ROS_TIME)

        # # 4. Calculate total E2E processing age (Sensor Capture -> Node Finished)
        # lidar_e2e_age_ms = (now - lidar_stamp).nanoseconds / 1e6
        # camera_e2e_age_ms = (now - camera_stamp).nanoseconds / 1e6

        # # 5. Measure the synchronization shift between the two sensor packets
        # sync_slop_ms = abs((lidar_stamp - camera_stamp).nanoseconds / 1e6)

        # # Print all metrics clearly categorized
        # self.get_logger().info(
        #     f"\n--- Fusion PERFORMANCE PROFILE ---\n"
        #     f"Active Objects     : 3D: {num_3d} | 2D: {num_2d}\n"
        #     f"Node Callback Cost : GPU Math: {gpu_time_ms:.2f}ms | Total Callback: {callback_time_ms:.2f}ms\n"
        #     f"Synchronizer Slop  : {sync_slop_ms:.2f}ms\n"
        #     f"End-to-End Stack Age: LiDAR E2E: {lidar_e2e_age_ms:.2f}ms | Camera E2E: {camera_e2e_age_ms:.2f}ms"
        # )

    def box_class_det(self, sx, sy, horizontal_footprint_area, aspect_ratio, inst_hypothesis, camera_class, score):
        # --- 3D BOUNDING BOX VOLUME SANITY FILTERS ---
        # Standard KITTI Volumetric Limits:
        # Pedestrian Volume: ~0.4m x 0.4m x 1.7m = ~0.3 m³ (Rarely exceeds 1.0 m³)
        # Car Volume: ~4.0m x 1.6m x 1.5m = ~9.6 m³ (Rarely drops below 3.5 m³)

        # --- ADVANCED CYCLIST VS CAR ANTI-MIXING FILTERS ---
        # CASE A: Camera claims CAR (0), but it is physically a CYCLIST
        if camera_class == 0:
            # If the cluster width is extremely narrow (< 0.85m) or the total 
            # footprint is tiny, it physically CANNOT be a car engine box.
            if sy < 0.85 or horizontal_footprint_area < 1.5:
                # If it is narrow and elongated, it is a Cyclist [1]
                if aspect_ratio >= 1.5:
                    camera_class = 2  # Override to Cyclist
                else:
                    camera_class = 1  # Override to Pedestrian (square-ish footprint)

        # CASE B: Camera claims CYCLIST (2), but it is physically a CAR
        elif camera_class == 2:
            # A cyclist is never 3.5+ meters long or wider than 1.1 meters.
            # If the horizontal footprint area is huge, it is a vehicle platform.
            if horizontal_footprint_area > 2.8 or sx > 2.8 or sy > 1.1:
                camera_class = 0  # Override to Car

        # CASE C: Camera claims PEDESTRIAN (1), but it is a CAR or CYCLIST
        elif camera_class == 1:
            if horizontal_footprint_area > 2.8 or sx > 2.8:
                camera_class = 0  # Override to Car
            elif sy < 0.85 and aspect_ratio >= 1.5:
                camera_class = 2  # Override to Cyclist

        # Write clean primitive string values out to the destination msg layout
        if hasattr(inst_hypothesis, 'hypothesis'):
            inst_hypothesis.hypothesis.class_id = str(camera_class)
            inst_hypothesis.hypothesis.score = score
        else:
            inst_hypothesis.class_id = str(camera_class)
            inst_hypothesis.score = float(score)

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
