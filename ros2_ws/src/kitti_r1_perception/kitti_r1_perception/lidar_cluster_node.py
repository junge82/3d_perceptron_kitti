import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Pose, PoseWithCovariance 
import open3d as o3d
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')

        # CRITICAL FIX: Declare arrays at the absolute start of __init__
        self.perf_log_data_age = []
        self.perf_log_ransac = []
        self.perf_log_dbscan = []
        self.perf_log_total = []
        
        # Parameters
        self.declare_parameter('voxel_size', 0.15)  
        self.declare_parameter('ground_distance_threshold', 0.3)
        self.declare_parameter('cluster_tolerance', 0.6)
        self.declare_parameter('min_cluster_size', 20)
        self.declare_parameter('max_cluster_size', 1000)
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.ground_dist = self.get_parameter('ground_distance_threshold').value
        self.cluster_tol = self.get_parameter('cluster_tolerance').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.max_cluster = self.get_parameter('max_cluster_size').value

        # Define the real-time profile inside __init__
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create a Mutually Exclusive Callback Group for thread isolation
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # Subscription
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kitti/velodyne/pointcloud',
            self.listener_callback,
            lidar_qos,  
            callback_group=self.callback_group) 
        
        # Publishers
        self.cluster_pub = self.create_publisher(Detection3DArray, '/detection/lidar_clusters', lidar_qos)
        self.no_ground_pub = self.create_publisher(PointCloud2, '/lidar/points_no_ground', lidar_qos)
        
        # Configure native Open3D device context targeting your RTX 4060
        self.device = o3d.core.Device("CUDA:0")

        # Metrics lists for post-run distribution plotting
        self.ransac_latencies = []
        self.dbscan_latencies = []
        self.total_latencies = []
        self.get_logger().info('Multi-Threaded Lidar Cluster Node Initialized with RTX 4060 GPU Acceleration & Geometric Filtering.')

    def listener_callback(self, msg):
# Start total callback tracking timer
        node_start_time = time.perf_counter()

        # 1. Fetch current clock directly from the node instance
        now = self.get_clock().now()
        
        # 2. Parse the raw message stamp cleanly
        msg_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        
        # 3. CRITICAL: Force calculation using the same integer timeline
        delta_nanoseconds = now.nanoseconds - msg_stamp.nanoseconds
        data_age_ms = delta_nanoseconds / 1e6

        # --- FIX: Clamp the initial clock connection lag ---
        if data_age_ms < -100000.0 or data_age_ms > 100000.0:
            data_age_ms = 0.0  # Force first frame to 0.0ms instead of showing the time warp

        # 4. Fast Vectorized Zero-Copy Byte Conversion (CPU to NumPy)
        conversion_start = time.perf_counter()

        # 1. Fast Vectorized Zero-Copy Byte Conversion (CPU to NumPy)
        conversion_start = time.perf_counter()
        points = self.fast_pointcloud2_to_numpy(msg)
        conversion_time_ms = (time.perf_counter() - conversion_start) * 1000.0
        
        if points.size == 0:
            return
        
        # 2. Vectorized ROI Box Filter (X=Forward, Y=Lateral, Z=Height)
        roi_start = time.perf_counter()
        roi_mask = (
            (points[:, 0] >= 1.5)   & (points[:, 0] <= 50.0) &  # X is FORWARD (1.5m to 50m ahead)
            (points[:, 1] >= -25.0) & (points[:, 1] <= 25.0) &  # Y is LATERAL (25m Right to 25m Left)
            (points[:, 2] >= -1.8)  & (points[:, 2] <= 2.0)      # Z is VERTICAL Height
        )
        points = points[roi_mask]
        roi_time_ms = (time.perf_counter() - roi_start) * 1000.0
        
        if points.size == 0:
            return
        
        # ----------------------------------------------------------------------
        # ⚡ GPU OFFLOAD: Instantly map arrays to RTX 4060 VRAM
        # ----------------------------------------------------------------------
        gpu_upload_start = time.perf_counter()
        o3d_points = o3d.core.Tensor(points, dtype=o3d.core.float32, device=self.device)
        pcd_gpu = o3d.t.geometry.PointCloud(o3d_points)
        upload_time_ms = (time.perf_counter() - gpu_upload_start) * 1000.0

        # 3. GPU Voxel Downsampling
        pcd_down = pcd_gpu.voxel_down_sample(voxel_size=self.voxel_size)

        # 4. GPU RANSAC Ground Segmentation
        if len(pcd_down.point.positions) < self.min_cluster:
            return

        ransac_start = time.perf_counter()
        plane_model, inliers_indices = pcd_down.segment_plane(distance_threshold=self.ground_dist,
                                                              ransac_n=3,
                                                              num_iterations=30)
        
        # Use select_by_index() to match Int64 types natively
        pcd_no_ground = pcd_down.select_by_index(inliers_indices, invert=True)
        ransac_time_ms = (time.perf_counter() - ransac_start) * 1000.0
        self.ransac_latencies.append(ransac_time_ms)
        
        if len(pcd_no_ground.point.positions) == 0:
            return

        # 5. GPU Parallel DBSCAN Clustering
        dbscan_start = time.perf_counter()
        labels_tensor = pcd_no_ground.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster)
        dbscan_time_ms = (time.perf_counter() - dbscan_start) * 1000.0
        self.dbscan_latencies.append(dbscan_time_ms)
        
        # ----------------------------------------------------------------------
        # HOST RETRIEVAL: Pull results back to CPU for ROS layout packaging
        # ----------------------------------------------------------------------
        labels = labels_tensor.cpu().numpy()
        no_ground_pts = pcd_no_ground.point.positions.cpu().numpy()

        max_label = labels.max() if labels.size > 0 else -1
        if max_label < 0:
            return

        # 6. Parallel Vector Center/Extent Processing via Numpy
        loop_start = time.perf_counter()
        detections = Detection3DArray()
        detections.header = msg.header
        if not detections.header.frame_id:
            detections.header.frame_id = "velodyne"
        
        for i in range(max_label + 1):
            cluster_mask = (labels == i)
            cluster_points = no_ground_pts[cluster_mask]
            
            num_points = cluster_points.shape[0]
            if num_points < self.min_cluster or num_points > self.max_cluster:
                continue
                
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            center = (min_bound + max_bound) / 2.0
            extent = max_bound - min_bound
            
            # ------------------------------------------------------------------
            # --- INTEGRATED: GEOMETRIC REAL OBJECT FILTERS ---
            # ------------------------------------------------------------------
            # Filter 1: Maximum Distance (Ignore crowd noise further than 35 meters away)
            distance_from_car = np.linalg.norm(center[:2]) 
            if distance_from_car > 35.0:
                continue

            # Filter 2: Profile Verification (Target typical Car / SUV dimensions)
            length, width, height = extent[0], extent[1], extent[2]
            
            # Eliminate oversized objects (walls, fences, long hedges)
            if length > 6.5 or width > 3.0 or height > 2.5:
                continue
                
            # Eliminate tiny scatter noise (stray dust points, floating leaves)
            if length < 0.3 or width < 0.3 or height < 0.3:
                continue

            # Filter 3: Aspect Ratio Rules (Differentiate cars from flat walls or skinny poles)
            if (length > 4.0 and width < 0.5) or (width > 4.0 and length < 0.5):
                continue
            # ------------------------------------------------------------------
            
            detection = Detection3D()
            detection.header = detections.header

            hypothesis = ObjectHypothesisWithPose()
            hp = ObjectHypothesis()
            hp.class_id = f'cluster_{i}'
            hp.score = 1.0
            hypothesis.hypothesis = hp
            
            hypothesis.pose = PoseWithCovariance()
            hypothesis.pose.pose = Pose()
            hypothesis.pose.pose.position.x = float(center[0])
            hypothesis.pose.pose.position.y = float(center[1])
            hypothesis.pose.pose.position.z = float(center[2])
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            
            bbox = BoundingBox3D()
            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.position.z = float(center[2])
            bbox.center.orientation.w = 1.0
            bbox.size.x = float(extent[0])
            bbox.size.y = float(extent[1])
            bbox.size.z = float(extent[2])
            
            detection.bbox = bbox
            detections.detections.append(detection)

        msg_loop_time_ms = (time.perf_counter() - loop_start) * 1000.0

        # Publish results
        self.cluster_pub.publish(detections)
        
        # Publish Debug Cloud
        no_ground_msg = point_cloud2.create_cloud_xyz32(detections.header, no_ground_pts)
        self.no_ground_pub.publish(no_ground_msg)

        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        num_clusters = len(detections.detections)

        # ----------------------------------------------------------------------
        # ⚡ CRITICAL OPTIMIZATION: Append metrics HERE, before any subsequent early returns
        # ----------------------------------------------------------------------
        self.perf_log_data_age.append(data_age_ms)
        self.perf_log_ransac.append(ransac_time_ms)
        self.perf_log_dbscan.append(dbscan_time_ms)
        self.perf_log_total.append(node_time_ms)
        # ----------------------------------------------------------------------


        # Log complete GPU performance metrics profile
        self.get_logger().info(
            f"\n--- RTX 4060 GPU LIDAR PROFILING BREAKDOWN ---\n"
            f"Generated Clusters  : {num_clusters}\n"
            f"Pre-Callback Age    : {data_age_ms:.2f}ms\n"
            f"Numpy Byte Slice    : {conversion_time_ms:.2f}ms\n"
            f"NumPy ROI Filter    : {roi_time_ms:.2f}ms\n"
            f"GPU VRAM Upload Cost: {upload_time_ms:.2f}ms\n"
            f"GPU CUDA RANSAC Cost: {ransac_time_ms:.2f}ms\n"
            f"GPU CUDA DBSCAN Cost: {dbscan_time_ms:.2f}ms\n"
            f"ROS Message Loop    : {msg_loop_time_ms:.2f}ms\n"
            f"Total Callback Cost : {node_time_ms:.2f}ms"
        )

    def save_latency_distribution_plot(self):
        """ Compiles a high-density diagnostic histogram with robust attribute fallback guards """
        # Safely extract arrays using getattr to completely immunize against AttributeError
        perf_log_total = getattr(self, 'perf_log_total', [])
        perf_log_data_age = getattr(self, 'perf_log_data_age', [])
        perf_log_ransac = getattr(self, 'perf_log_ransac', [])
        perf_log_dbscan = getattr(self, 'perf_log_dbscan', [])

        if not perf_log_total:
            self.get_logger().warning("No performance metrics logged or initialization skipped. Bypassing graph generation.")
            return

        try:
            # Resolve paths
            pkg_share = get_package_share_directory('kitti_r1_perception')
            log_dir = os.path.join(pkg_share, 'perf_logs')
            os.makedirs(log_dir, exist_ok=True)
            plot_path = os.path.join(log_dir, 'lidar_latency_distribution.png')

            # Create a 2x2 grid panel plot
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('RTX 4060 LiDAR Pipeline Latency Analysis', fontsize=16, fontweight='bold')

            # Plot 1: Pre-Callback Transport Data Age
            axs[0, 0].hist(perf_log_data_age, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axs[0, 0].set_title('Pre-Callback Data Age (Network Transport Delay)')
            axs[0, 0].set_xlabel('Latency (ms)')
            axs[0, 0].set_ylabel('Frame Count')

            # Plot 2: GPU RANSAC Ground Segmentation Execution Cost
            axs[0, 1].hist(perf_log_ransac, bins=20, color='cyan', alpha=0.7, edgecolor='black')
            axs[0, 1].set_title('GPU CUDA RANSAC Cost')
            axs[0, 1].set_xlabel('Latency (ms)')
            axs[0, 1].set_ylabel('Frame Count')

            # Plot 3: GPU DBSCAN Clustering Execution Cost
            axs[1, 0].hist(perf_log_dbscan, bins=20, color='purple', alpha=0.7, edgecolor='black')
            axs[1, 0].set_title('GPU CUDA DBSCAN Clustering Cost')
            axs[1, 0].set_xlabel('Latency (ms)')
            axs[1, 0].set_ylabel('Frame Count')

            # Plot 4: Total Node Callback Processing Footprint
            axs[1, 1].hist(perf_log_total, bins=20, color='green', alpha=0.7, edgecolor='black')
            axs[1, 1].set_title('Total Node Callback Footprint')
            axs[1, 1].set_xlabel('Latency (ms)')
            axs[1, 1].set_ylabel('Frame Count')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"\n[PERF DATA LOG] Latency distribution graph successfully saved to: {plot_path}\n")

        except Exception as e:
            print(f"Failed to generate performance plot: {e}")

    def fast_pointcloud2_to_numpy(self, msg):
        """ Zero-copy memory view slicing. Skips generators entirely. """
        data_arr = np.frombuffer(msg.data, dtype=np.uint8)
        point_step = msg.point_step
        num_points = (msg.height * msg.width)
        reshaped = data_arr.reshape(num_points, point_step)
        
        x_pts = reshaped[:, 0:4].view(dtype=np.float32)
        y_pts = reshaped[:, 4:8].view(dtype=np.float32)
        z_pts = reshaped[:, 8:12].view(dtype=np.float32)
        
        points = np.hstack((x_pts, y_pts, z_pts))
        return points[np.isfinite(points).all(axis=1)]

def main(args=None):
    rclpy.init(args=args)

    try:
        node = LidarClusterNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Trigger the multi-stage performance plot on shutdown
        node.save_latency_distribution_plot()
        node.destroy_node()

        # FIXED: Enforce clear context protection guard for ROS 2 Jazzy
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
