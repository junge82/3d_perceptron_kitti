import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor  # Use a single-threaded executor for predictable processing.
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

        self.perf_log_data_age = []
        self.perf_log_ransac = []
        self.perf_log_dbscan = []
        self.perf_log_total = []
        
        # --- Configuration ---
        self.declare_parameter('voxel_size', 0.15)  
        self.declare_parameter('ground_distance_threshold', 0.3)
        self.declare_parameter('ransac_n', 3)
        self.declare_parameter('ransac_iterations', 30)
        self.declare_parameter('cluster_tolerance', 0.6)
        self.declare_parameter('min_cluster_size', 20)
        self.declare_parameter('max_cluster_size', 1000)
        self.declare_parameter('bev_min_x', 1.5)
        self.declare_parameter('bev_max_x', 50.0)
        self.declare_parameter('bev_min_y', -25.0)
        self.declare_parameter('bev_max_y', 25.0)
        self.declare_parameter('bev_min_z', -1.8)
        self.declare_parameter('bev_max_z', 2.0)
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.ground_dist = self.get_parameter('ground_distance_threshold').value
        self.ransac_n = self.get_parameter('ransac_n').value
        self.ransac_iterations = self.get_parameter('ransac_iterations').value
        self.cluster_tol = self.get_parameter('cluster_tolerance').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.max_cluster = self.get_parameter('max_cluster_size').value
        self.bev_min_x = self.get_parameter('bev_min_x').value
        self.bev_max_x = self.get_parameter('bev_max_x').value
        self.bev_min_y = self.get_parameter('bev_min_y').value
        self.bev_max_y = self.get_parameter('bev_max_y').value
        self.bev_min_z = self.get_parameter('bev_min_z').value
        self.bev_max_z = self.get_parameter('bev_max_z').value

        # Configure a bounded QoS profile for point-cloud processing.
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Prevent concurrent execution of callbacks in this node.
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # --- ROS interfaces ---
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kitti/velodyne/pointcloud',
            self.listener_callback,
            lidar_qos,  
            callback_group=self.callback_group) 
        
        # Publish clustered detections and the ground-filtered point cloud.
        self.cluster_pub = self.create_publisher(Detection3DArray, '/detection/lidar_clusters', lidar_qos)
        self.no_ground_pub = self.create_publisher(PointCloud2, '/lidar/points_no_ground', lidar_qos)
        
        # Use the GPU when supported by the installed Open3D build; otherwise use the CPU.
        try:
            self.o3d_device = o3d.core.Device("CUDA:0")
            self.get_logger().info("Successfully bound Open3D to RTX 4060 GPU.")
        except Exception:
            self.o3d_device = o3d.core.Device("CPU:0")
            self.get_logger().warn("CUDA device not supported by Open3D build. Falling back to CPU mode.")

        self.get_logger().info('Lidar Cluster Node Initialized successfully.')

        # Warm up the CPU and Open3D processing paths.
        self.warmup_gpu()

        # Initialize metrics used by the post-run latency plots.
        self.ransac_latencies = []
        self.dbscan_latencies = []
        self.total_latencies = []
        self.get_logger().info('Multi-Threaded Lidar Cluster Node Initialized with RTX 4060 GPU Acceleration & Geometric Filtering.')

    def warmup_gpu(self):
        """Warm up point-cloud processing with representative sensor data."""
        try:
            # Use a representative KITTI point-cloud size.
            exact_sensor_shape = (115000, 3)
            dummy_pts = np.random.uniform(-10.0, 40.0, exact_sensor_shape).astype(np.float32)
            
            # Apply the same region-of-interest filter used for live data.
            dummy_pts = dummy_pts[self.get_roi_mask(dummy_pts)]

            # Create an Open3D point cloud on the selected device.
            pcd_gpu = o3d.t.geometry.PointCloud(device=self.o3d_device)
            pcd_gpu.point["positions"] = o3d.core.Tensor(dummy_pts, dtype=o3d.core.float32, device=self.o3d_device)

            # Run the same downsampling, ground segmentation, and clustering stages as the live pipeline.
            pcd_down = pcd_gpu.voxel_down_sample(voxel_size=self.voxel_size)
            _, inliers = pcd_down.segment_plane(
                self.ground_dist, self.ransac_n, self.ransac_iterations
            )

            pcd_no_ground = pcd_down.select_by_index(inliers, invert=True)
            _ = pcd_no_ground.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster)

            self.get_logger().info('CUDA Pipelines & NumPy ROI structures compiled successfully.')
        except Exception as e:
            self.get_logger().warn(f'Warmup execution bypassed: {str(e)}')

    def get_roi_mask(self, points):
        return (
            (points[:, 0] >= self.bev_min_x) & (points[:, 0] <= self.bev_max_x) &
            (points[:, 1] >= self.bev_min_y) & (points[:, 1] <= self.bev_max_y) &
            (points[:, 2] >= self.bev_min_z) & (points[:, 2] <= self.bev_max_z)
        )

    def listener_callback(self, msg):
        # --- Input timing and preprocessing ---
        node_start_time = time.perf_counter()

        # Measure message age using the node's ROS clock.
        now = self.get_clock().now()
        
        msg_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        
        delta_nanoseconds = now.nanoseconds - msg_stamp.nanoseconds
        data_age_ms = delta_nanoseconds / 1e6

        # Ignore extreme values during initial clock synchronization.
        if data_age_ms < -100000.0 or data_age_ms > 100000.0:
            data_age_ms = 0.0  # Avoid reporting the initial clock offset as message age.

        # Convert the point-cloud buffer to a NumPy view.
        conversion_start = time.perf_counter()
        points = self.fast_pointcloud2_to_numpy(msg)
        conversion_time_ms = (time.perf_counter() - conversion_start) * 1000.0
        
        if points.size == 0:
            return
        
        # Filter points to the configured BEV region of interest.
        roi_start = time.perf_counter()
        points = points[self.get_roi_mask(points)]
        roi_time_ms = (time.perf_counter() - roi_start) * 1000.0
        if points.size == 0:
            return
        
        # --- Open3D processing ---
        # Transfer points to the selected Open3D device.
        gpu_upload_start = time.perf_counter()
        o3d_points = o3d.core.Tensor(points, dtype=o3d.core.float32, device=self.o3d_device)
        pcd_gpu = o3d.t.geometry.PointCloud(device=self.o3d_device)
        pcd_gpu.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.float32, device=self.o3d_device)
        upload_time_ms = (time.perf_counter() - gpu_upload_start) * 1000.0

        # Reduce point density with voxel downsampling.
        pcd_down = pcd_gpu.voxel_down_sample(voxel_size=self.voxel_size)

        # Segment the ground plane with RANSAC.
        if len(pcd_down.point.positions) < self.min_cluster:
            return

        ransac_start = time.perf_counter()
        plane_model, inliers_indices = pcd_down.segment_plane(distance_threshold=self.ground_dist,
                                                              ransac_n=self.ransac_n,
                                                              num_iterations=self.ransac_iterations)
        
        # Remove ground points using the returned inlier indices.
        pcd_no_ground = pcd_down.select_by_index(inliers_indices, invert=True)
        ransac_time_ms = (time.perf_counter() - ransac_start) * 1000.0
        self.ransac_latencies.append(ransac_time_ms)
        
        if len(pcd_no_ground.point.positions) == 0:
            return

        # Bound the number of points passed to DBSCAN.
        MAX_GPU_COMPUTE_POINTS = 15000
        num_noground_points = len(pcd_no_ground.point.positions)

        if num_noground_points > MAX_GPU_COMPUTE_POINTS:            
            # Select a bounded subset of points on the active Open3D device.
            indices_np = np.arange(MAX_GPU_COMPUTE_POINTS, dtype=np.int64)
            indices = o3d.core.Tensor(
                indices_np,
                dtype=o3d.core.Dtype.Int64,
                device=self.o3d_device,
            )            
            pcd_no_ground = pcd_no_ground.select_by_index(indices)
            
            self.get_logger().debug(
                f"[LATENCY GUARD] Clipped total unclustered points from {num_noground_points} "
                f"to memory allocation ceiling {MAX_GPU_COMPUTE_POINTS} to block VRAM pool overflows."
            )

        # Cluster the remaining points with DBSCAN.
        dbscan_start = time.perf_counter()
        labels_tensor = pcd_no_ground.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster)
        dbscan_time_ms = (time.perf_counter() - dbscan_start) * 1000.0
        self.dbscan_latencies.append(dbscan_time_ms)
        
        # --- ROS message construction ---
        # Copy clustering results to CPU for NumPy processing and ROS serialization.
        labels = labels_tensor.cpu().numpy()
        no_ground_pts = pcd_no_ground.point.positions.cpu().numpy()

        max_label = labels.max() if labels.size > 0 else -1
        if max_label < 0:
            return

        # Compute cluster bounds and construct detection messages.
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

            # --- Geometry-based cluster filtering ---
            
            # Reject clusters beyond the supported detection range.
            distance_from_car = float(np.linalg.norm(center[:2]))
            if distance_from_car > 35.0:
                continue

            length, width, height = float(extent[0]), float(extent[1]), float(extent[2])
            horizontal_area = length * width
            volume = length * width * height

            # Identify long structures near the lateral edges of the road.
            is_lateral_infrastructure = (abs(center[1]) > 3.5) and (length > 5.5)

            # Reject dimensions beyond the supported vehicle and infrastructure bounds.
            
            if length > 45.0 or width > 12.0 or height > 5.5:
                continue

            # Apply stricter filtering nearby and size-based filtering at longer range.
            if distance_from_car <= 15.0:
                if volume < 0.05 or height < 0.2:
                    continue
                
                # Reject lateral infrastructure in the nearby region.
                if is_lateral_infrastructure:
                    continue

                aspect_ratio = max(length, width) / max(min(length, width), 0.001)
                if aspect_ratio > 9.0 and max(length, width) > 5.0 and abs(center[1]) > 2.0:
                    continue
                    
            else:
                # Apply minimum dimensions in the mid- and far-range region.
                if length < 0.4 or width < 0.4 or height < 0.4:
                    continue
                
                if horizontal_area < 0.20 or volume < 0.10:  # Allow smaller clusters at longer range.
                    continue

                # Reject only extremely thin, elongated structures at longer range.
                if (length > 5.0 and width < 0.4) or (width > 5.0 and length < 0.4):
                    continue

            # --- Detection message ---
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

            # Map the cluster dimensions to the corresponding bounding-box axes.
            bbox = BoundingBox3D()
            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.position.z = float(center[2])
            bbox.center.orientation.w = 1.0

            bbox.size.x = float(length)  # Length along the x-axis.
            bbox.size.y = float(width)   # Width along the y-axis.
            bbox.size.z = float(height)  # Height along the z-axis.

            detection.bbox = bbox
            detections.detections.append(detection)

        msg_loop_time_ms = (time.perf_counter() - loop_start) * 1000.0

        # --- Publish outputs and record metrics ---
        self.cluster_pub.publish(detections)

        # Publish the ground-filtered point cloud for visualization.
        no_ground_msg = point_cloud2.create_cloud_xyz32(detections.header, no_ground_pts)
        self.no_ground_pub.publish(no_ground_msg)

        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        num_clusters = len(detections.detections)

        # Store callback metrics for the post-run plots.
        self.perf_log_data_age.append(data_age_ms)
        self.perf_log_ransac.append(ransac_time_ms)
        self.perf_log_dbscan.append(dbscan_time_ms)
        self.perf_log_total.append(node_time_ms)
        # Log a breakdown of the callback processing time.
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
        """Save latency histograms for the LiDAR processing stages."""
        # Read available metric histories.
        perf_log_total = getattr(self, 'perf_log_total', [])
        perf_log_data_age = getattr(self, 'perf_log_data_age', [])
        perf_log_ransac = getattr(self, 'perf_log_ransac', [])
        perf_log_dbscan = getattr(self, 'perf_log_dbscan', [])

        if not perf_log_total:
            self.get_logger().warning("No performance metrics logged or initialization skipped. Bypassing graph generation.")
            return

        try:
            # Resolve the package's performance log directory.
            pkg_share = get_package_share_directory('kitti_r1_perception')
            log_dir = os.path.join(pkg_share, 'perf_logs')
            os.makedirs(log_dir, exist_ok=True)
            plot_path = os.path.join(log_dir, 'lidar_latency_distribution.png')

            # Create a 2-by-2 grid of latency histograms.
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('RTX 4060 LiDAR Pipeline Latency Analysis', fontsize=16, fontweight='bold')

            # Plot 1: message age before callback processing.
            axs[0, 0].hist(perf_log_data_age, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axs[0, 0].set_title('Pre-Callback Data Age (Network Transport Delay)')
            axs[0, 0].set_xlabel('Latency (ms)')
            axs[0, 0].set_ylabel('Frame Count')

            # Plot 2: RANSAC ground-segmentation latency.
            axs[0, 1].hist(perf_log_ransac, bins=20, color='cyan', alpha=0.7, edgecolor='black')
            axs[0, 1].set_title('GPU CUDA RANSAC Cost')
            axs[0, 1].set_xlabel('Latency (ms)')
            axs[0, 1].set_ylabel('Frame Count')

            # Plot 3: DBSCAN clustering latency.
            axs[1, 0].hist(perf_log_dbscan, bins=20, color='purple', alpha=0.7, edgecolor='black')
            axs[1, 0].set_title('GPU CUDA DBSCAN Clustering Cost')
            axs[1, 0].set_xlabel('Latency (ms)')
            axs[1, 0].set_ylabel('Frame Count')

            # Plot 4: total node callback latency.
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
        """Return XYZ points as a zero-copy view of the PointCloud2 data buffer."""
        if len(msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # View the message payload as a 1D uint8 array.
        data_arr = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape the buffer into one row per point.
        reshaped = data_arr.reshape((msg.height * msg.width), msg.point_step)
        
        # View the first 12 bytes of each point as three consecutive float32 XYZ values.
        points = reshaped[:, 0:12].view(dtype=np.float32).reshape(-1, 3)
          
        return points

def main(args=None):
    rclpy.init(args=args)

    try:
        node = LidarClusterNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Save latency plots before shutting down the node.
        node.save_latency_distribution_plot()
        node.destroy_node()

        # Shut down the ROS context only if it is still active.
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
