import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from vision_msgs.msg import Detection3DArray, Detection2DArray, ObjectHypothesisWithPose
import numpy as np
import message_filters
import torch 
import time  # High-resolution CPU timer.

from kitti_r1_perception.calibration import K_KITTI, T_LIDAR_TO_CAM


class FrustumFusionNode(Node):
    def __init__(self):
        super().__init__('frustum_fusion_node')

        # --- Initialization ---
        # Initialize callback state before registering subscriptions.
        self.latest_camera_msg = None

        # Enable line-buffered output for timely log display.
        import sys
        sys.stdout.reconfigure(line_buffering=True)

        # --- Fusion parameters ---
        self.declare_parameter('match_radius', 50.0)  # Maximum 2D association distance in pixels.
        self.match_radius = self.get_parameter('match_radius').value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Create CUDA events for GPU timing when CUDA is available.
        self.is_cuda = self.device.type == 'cuda'
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        # --- KITTI calibration ---
        self.K_tensor = torch.from_numpy(K_KITTI).to(self.device)
        self.R_tensor = torch.from_numpy(T_LIDAR_TO_CAM[:3, :3]).to(self.device)
        self.t_tensor = torch.from_numpy(T_LIDAR_TO_CAM[:3, 3]).to(self.device)

        # Use reliable delivery and a bounded queue for synchronized inputs.
        optimized_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5  # Buffer a small number of messages for synchronization.
        )

        # Ensure camera and LiDAR subscription callbacks do not run concurrently.
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # --- ROS interfaces ---
        # Publish fused 3D detections.
        self.fused_pub = self.create_publisher(Detection3DArray, '/detection/fused_3d', 2)

        # Subscribe to camera and LiDAR detections for timestamp-based pairing.
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

        # Allow a small timestamp difference between sensor detections.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_filter, self.lidar_filter],
            queue_size=3,  # Keep the synchronization queue bounded.
            slop=0.03,  # Allow up to 30 ms of timestamp difference.
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Vectorized CUDA Frustum Fusion Engine Activated Successfully.")

    def extract_centers(self, msg, is_3d: bool = False) -> np.ndarray:
        # Extracts detection bounding box centers into an (N, 2) or (N, 3) NumPy array."""
        cols = 3 if is_3d else 2
        if not msg.detections:
            return np.empty((0, cols), dtype=np.float32)

        pts = [self.parse_single_center(det.bbox.center, is_3d) for det in msg.detections]
        return np.array(pts, dtype=np.float32)

    def parse_single_center(self, center, is_3d: bool) -> list[float]:
        # Validates and extracts coordinates from a single bounding box center."""
        # Support centers represented as either a nested position or direct coordinates.
        src = center.position if hasattr(center, 'position') else center

        if not (hasattr(src, 'x') and hasattr(src, 'y')):
            raise ValueError(f"Unknown center schema attributes: {dir(center)}")

        if is_3d:
            if not hasattr(src, 'z'):
                raise ValueError(f"3D coordinates requested, but 'z' attribute is missing: {dir(center)}")
            return [float(src.x), float(src.y), float(src.z)]

        return [float(src.x), float(src.y)]

    def sync_callback(self, camera_msg, lidar_msg):
        # --- Callback timing and input validation ---
        callback_start_time = time.perf_counter()        

        # Skip processing if a camera message is unavailable.
        if camera_msg is None:
            self.get_logger().info(
                "Fusion Layer Status: [WARMUP] Waiting for camera msgs...",
                throttle_duration_sec=1.5  # Limit repeated warmup messages.
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

        # --- Extract detection coordinates ---
        try:
            lidar_pts_np = self.extract_centers(lidar_msg, is_3d=True)
            camera_pts_np = self.extract_centers(camera_msg, is_3d=False)

            
        except Exception as e:
            self.get_logger().error(f"🚨 Structure parsing exception: {str(e)}")
            return

        pts_3d_lidar = torch.as_tensor(lidar_pts_np, device=self.device)
        pts_2d_tensor = torch.as_tensor(camera_pts_np, device=self.device)

        # --- Project LiDAR centers into the camera image ---
        # Transform LiDAR points into the camera coordinate frame.
        pts_3d_cam = torch.addmm(self.t_tensor, pts_3d_lidar, self.R_tensor.t())

        # Remove points behind or too close to the camera.
        valid_mask = pts_3d_cam[:, 2] > 0.1
        if not torch.any(valid_mask):
            self.get_logger().info("Sync Match: No valid 3D points in camera frame, skipping frame.")
            return

        # Retain the indices of points in front of the camera.
        depth_valid_indices = torch.where(valid_mask)[0]
        pts_3d_cam_filtered = pts_3d_cam[depth_valid_indices]

        # Prepare a zero bias matrix with one row per valid point.
        num_valid_pts = pts_3d_cam_filtered.shape[0]
        bias_tensor = torch.zeros((num_valid_pts, 3), dtype=torch.float32, device=self.device)
        
        try:
            proj_pts = torch.addmm(bias_tensor, pts_3d_cam_filtered, self.K_tensor.t())
            
            # Clamp depth to avoid division by zero and invalid projections.
            z_clamped = torch.clamp(proj_pts[:, 2:3], min=1e-4)
            inv_z = 1.0 / z_clamped
            
            # Convert projected points to image pixel coordinates.
            uv_gpu = proj_pts[:, :2] * inv_z

            # Retain nearby objects that extend beyond the image boundaries.
            # Extend the horizontal and vertical bounds by 200 and 100 pixels.
            PADDING_X = 200.0
            PADDING_Y = 100.0

            in_fov_mask = (uv_gpu[:, 0] >= -PADDING_X) & (uv_gpu[:, 0] <= (1248.0 + PADDING_X)) & \
                          (uv_gpu[:, 1] >= -PADDING_Y) & (uv_gpu[:, 1] <= (384.0 + PADDING_Y))

            # Keep projections within the buffered camera field of view.
            fov_indices = torch.where(in_fov_mask)[0]
            uv_gpu_clipped = uv_gpu[fov_indices]
            
        except RuntimeError as e:
            self.get_logger().error(f"🚨 Matrix Multiplication Shape Mismatch inside Fusion Layer: {str(e)}")
            return
        except Exception as e:
            self.get_logger().error(f"🚨 Projection Execution Exception: {str(e)}")
            return

        # --- Optional projection diagnostics ---
        # Inspect the first valid projected point.
        if uv_gpu_clipped.shape[0] > 0:
            first_u = float(uv_gpu_clipped[0, 0].item())
            first_v = float(uv_gpu_clipped[0, 1].item())
            
            # Log whether the projected point falls inside the image bounds.
            self.get_logger().info(
                f"[PROJECTION ALIGNMENT CHECK] Projected 2D Pixel: (u={first_u:.1f}, v={first_v:.1f}) | "
                f"Valid 1248x384 Range: {0.0 <= first_u <= 1248.0 and 0.0 <= first_v <= 384.0}",
                throttle_duration_sec=2.0
            )

        # --- Associate projected points with camera detections ---
        # Compute pairwise pixel distances using tensor broadcasting.
        diff = uv_gpu_clipped.unsqueeze(1) - pts_2d_tensor.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=2) 
        min_dists, best_2d_indices = torch.min(dist_matrix, dim=1)

        # Copy association results to CPU for message construction.
        min_dists_cpu = min_dists.cpu().numpy()
        best_2d_indices_cpu = best_2d_indices.cpu().numpy()
        
        # Flatten the valid indices to preserve one-dimensional indexing.
        valid_indices_cpu = depth_valid_indices[fov_indices].cpu().numpy().flatten()

        # --- End of timed computation ---
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()  
            compute_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            compute_time_ms = (time.perf_counter() - gpu_start_cpu_fallback) * 1000.0
        # --- Build fused detections ---
        fused_detections = Detection3DArray()
        fused_detections.header = lidar_msg.header

        
        self.fuse_lidar_and_camera_detections(camera_msg, lidar_msg, min_dists_cpu, best_2d_indices_cpu, valid_indices_cpu, fused_detections)

        # Publish only when at least one detection was fused.
        if len(fused_detections.detections) > 0:
            self.fused_pub.publish(fused_detections)

        lidar_stamp_ns = lidar_msg.header.stamp.sec * 1_000_000_000 + lidar_msg.header.stamp.nanosec
        camera_stamp_ns = camera_msg.header.stamp.sec * 1_000_000_000 + camera_msg.header.stamp.nanosec
        sync_slop_ms = abs(lidar_stamp_ns - camera_stamp_ns) / 1e6
        callback_time_ms = (time.perf_counter() - callback_start_time) * 1000.0

        self.get_logger().info(
            f"\n--- Frustum Fusion Callback ---\n"
            f"Input clusters (LiDAR / camera): {num_3d} / {num_2d}\n"
            f"Fused detections              : {len(fused_detections.detections)}\n"
            f"Timestamp difference          : {sync_slop_ms:.2f} ms\n"
            f"Compute interval              : {compute_time_ms:.2f} ms\n"
            f"Total callback                : {callback_time_ms:.2f} ms\n"
            f"--------------------------------"
        )

    def fuse_lidar_and_camera_detections(self, camera_msg, lidar_msg, min_dists_cpu, best_2d_indices_cpu, valid_indices_cpu, fused_detections):
        for i, lidar_idx in enumerate(valid_indices_cpu):
            det_3d = lidar_msg.detections[lidar_idx]

            # Compute the cluster distance from the sensor origin.
            cx = float(det_3d.bbox.center.position.x)
            cy = float(det_3d.bbox.center.position.y)
            distance_meters = np.hypot(cx, cy)
            
            # Start with the configured association radius.
            current_match_radius = self.match_radius
            
            # Increase the radius for nearby objects to accommodate larger image distortions.
            if distance_meters <= 10.0:
                current_match_radius = self.match_radius * 3.0

            if min_dists_cpu[i] < current_match_radius:
                best_match_2d = camera_msg.detections[best_2d_indices_cpu[i]]

                # Read the LiDAR box dimensions.
                sx = det_3d.bbox.size.x  # Length along the longitudinal axis.
                sy = det_3d.bbox.size.y  # Width along the lateral axis.
                sz = det_3d.bbox.size.z  # Height along the vertical axis.
                
                # Compute geometric features for class correction.
                horizontal_footprint_area = sx * sy
                aspect_ratio = max(sx, sy) / max(min(sx, sy), 0.001)  # Length-to-width ratio.
                
                # Replace any existing class hypotheses.
                det_3d.results = []

                for res_2d in best_match_2d.results:
                    inst_hypothesis = ObjectHypothesisWithPose()
                    
                    # Read the camera class ID and confidence across supported message layouts.
                    if hasattr(res_2d, 'hypothesis'):
                        camera_class = int(res_2d.hypothesis.class_id)
                        score = float(res_2d.hypothesis.score)
                    else:
                        camera_class = int(res_2d.class_id)
                        score = res_2d.score

                    self.box_class_det(sx, sy, horizontal_footprint_area, aspect_ratio, inst_hypothesis, camera_class, score)
                        
                    det_3d.results.append(inst_hypothesis)

                # Add the fused detection to the output message.
                fused_detections.detections.append(det_3d)

    def box_class_det(self, sx, sy, horizontal_footprint_area, aspect_ratio, inst_hypothesis, camera_class, score):
        # --- Geometry-based class correction ---
        # Apply approximate KITTI object dimensions as class-consistency heuristics.

        # Reclassify camera predictions when the LiDAR geometry strongly disagrees.
        # Case A: The camera predicts a car, but the footprint is too small.
        if camera_class == 0:
            # A narrow or small footprint is unlikely to belong to a car.
            if sy < 0.85 or horizontal_footprint_area < 1.5:
                # An elongated footprint is more consistent with a cyclist.
                if aspect_ratio >= 1.5:
                    camera_class = 2  # Reclassify as a cyclist.
                else:
                    camera_class = 1  # Reclassify as a pedestrian.

        # Case B: The camera predicts a cyclist, but the footprint is vehicle-sized.
        elif camera_class == 2:
            # Large dimensions or footprint area are more consistent with a car.
            if horizontal_footprint_area > 2.8 or sx > 2.8 or sy > 1.1:
                camera_class = 0  # Reclassify as a car.

        # Case C: The camera predicts a pedestrian, but the footprint suggests a vehicle.
        elif camera_class == 1:
            if horizontal_footprint_area > 2.8 or sx > 2.8:
                camera_class = 0  # Reclassify as a car.
            elif sy < 0.85 and aspect_ratio >= 1.5:
                camera_class = 2  # Reclassify as a cyclist.

        # Store class IDs and scores in the supported message layout.
        if hasattr(inst_hypothesis, 'hypothesis'):
            inst_hypothesis.hypothesis.class_id = str(camera_class)
            inst_hypothesis.hypothesis.score = score
        else:
            inst_hypothesis.class_id = str(camera_class)
            inst_hypothesis.score = float(score)

def main(args=None):
    rclpy.init(args=args)
    node = FrustumFusionNode()

    # --- Executor lifecycle ---
    executor = MultiThreadedExecutor(num_threads=2)  # Use two threads for executor callbacks.
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        
        # Shut down the ROS context only if it is still active.
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
