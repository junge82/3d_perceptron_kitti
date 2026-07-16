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
import time  # High-precision CPU timer for profiling

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')
        
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
        
        self.get_logger().info('Multi-Threaded Lidar Cluster Node Initialized with Split Profiling.')

    def listener_callback(self, msg):
        node_start_time = time.perf_counter()

        # Extract current simulation timeline clock
        now = self.get_clock().now()
        msg_stamp = rclpy.time.Time.from_msg(msg.header.stamp, clock_type=rclpy.time.ClockType.ROS_TIME)
        data_age_ms = (now - msg_stamp).nanoseconds / 1e6

        # 1. Fast Vectorized Zero-Copy Byte Conversion
        points = self.fast_pointcloud2_to_numpy(msg)
        if points.size == 0:
            return
        
        # ----------------------------------------------------------------------
        # FIXED: Corrected KITTI ROI Window Filter (X=Forward, Y=Lateral, Z=Height)
        # ----------------------------------------------------------------------
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
        
        # 2. Setup Open3D and Downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 3. Fast RANSAC Ground Segmentation
        if len(pcd_down.points) < self.min_cluster:
            return

        ransac_start = time.perf_counter()
        plane_model, inliers = pcd_down.segment_plane(distance_threshold=self.ground_dist,
                                                     ransac_n=3,
                                                     num_iterations=30) # Dropped to 30 for speed
        
        pcd_no_ground = pcd_down.select_by_index(inliers, invert=True)
        no_ground_pts = np.asarray(pcd_no_ground.points)
        ransac_time_ms = (time.perf_counter() - ransac_start) * 1000.0
        
        if no_ground_pts.size == 0:
            return

        # 4. Euclidean DBSCAN Clustering (Now processing significantly fewer points)
        dbscan_start = time.perf_counter()
        labels = np.array(pcd_no_ground.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster, print_progress=False))
        dbscan_time_ms = (time.perf_counter() - dbscan_start) * 1000.0
        
        max_label = labels.max() if labels.size > 0 else -1
        if max_label < 0:
            return

        # 5. Parallel Vector Center/Extent Processing via Numpy
        loop_start = time.perf_counter()
        detections = Detection3DArray()
        detections.header = msg.header
        
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
            
            detection = Detection3D()
            detection.header = msg.header

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
        no_ground_msg = point_cloud2.create_cloud_xyz32(msg.header, no_ground_pts)
        self.no_ground_pub.publish(no_ground_msg)

        node_time_ms = (time.perf_counter() - node_start_time) * 1000.0
        num_clusters = len(detections.detections)

        # Log detailed execution breakdown
        self.get_logger().info(
            f"\n--- LIDAR PROFILING BREAKDOWN ---\n"
            f"Generated Clusters  : {num_clusters}\n"
            f"Pre-Callback Age    : {data_age_ms:.2f}ms\n"
            f"NumPy ROI Window Filter: {roi_time_ms:.2f}ms\n"
            f"RANSAC Ground Plane : {ransac_time_ms:.2f}ms\n"
            f"DBSCAN Clustering   : {dbscan_time_ms:.2f}ms\n"
            f"ROS Message Loop    : {msg_loop_time_ms:.2f}ms\n"
            f"Total Callback Cost : {node_time_ms:.2f}ms"
        )

    def fast_pointcloud2_to_numpy(self, msg):
        """ Zero-copy memory view slicing. Skips generators entirely. """
        # Extract raw byte buffer array
        data_arr = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Parse byte structural parameters
        row_step = msg.row_step
        point_step = msg.point_step
        
        # Reshape to easily slice fields out via item steps
        num_points = (msg.height * msg.width)
        reshaped = data_arr.reshape(num_points, point_step)
        
        # Pull offsets for x, y, z fields (Assuming typical float32 layouts: x=0, y=4, z=8)
        x_pts = reshaped[:, 0:4].view(dtype=np.float32)
        y_pts = reshaped[:, 4:8].view(dtype=np.float32)
        z_pts = reshaped[:, 8:12].view(dtype=np.float32)
        
        points = np.hstack((x_pts, y_pts, z_pts))
        return points[np.isfinite(points).all(axis=1)]

def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()

    # 4. Initialize the MultiThreadedExecutor
    # It automatically detects your CPU core count and provisions an optimal thread pool
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
