import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose, ObjectHypothesis
from geometry_msgs.msg import Pose, PoseWithCovariance 
import open3d as o3d
import numpy as np

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')
        
        # Parameters (configurable via CLI or launch)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('ground_distance_threshold', 0.5)
        self.declare_parameter('cluster_tolerance', 0.5)
        self.declare_parameter('min_cluster_size', 50)
        self.declare_parameter('max_cluster_size', 1000)
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.ground_dist = self.get_parameter('ground_distance_threshold').value
        self.cluster_tol = self.get_parameter('cluster_tolerance').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.max_cluster = self.get_parameter('max_cluster_size').value

        # Eszköz kiválasztása (GPU ha elérhető, egyébként CPU)
        self.device = o3d.core.Device("CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
        self.get_logger().info(f'Lidar Cluster Node indítása. Használt eszköz: {self.device}')

        # Subscription
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kitti/velodyne/pointcloud',
            self.listener_callback,
            10)
        
        # Publishers
        self.cluster_pub = self.create_publisher(Detection3DArray, '/detection/lidar_clusters', 10)
        self.no_ground_pub = self.create_publisher(PointCloud2, '/lidar/points_no_ground', 10)

    def listener_callback(self, msg):
        # Live-update parameters
        self.voxel_size = self.get_parameter('voxel_size').value
        self.ground_dist = self.get_parameter('ground_distance_threshold').value
        self.cluster_tol = self.get_parameter('cluster_tolerance').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.max_cluster = self.get_parameter('max_cluster_size').value

        # 1. Convert ROS PointCloud2 to NumPy
        points = self.pointcloud2_to_numpy(msg)
        if points.size == 0:
            return

        # 2. Helyes Tensor alapú (t) PointCloud létrehozása a kiválasztott eszközön (GPU/CPU)
        points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32, device=self.device)
        pcd = o3d.t.geometry.PointCloud(self.device)
        pcd.point["positions"] = points_tensor

        # 3. Downsample a GPU-n (Tensor API)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Biztonsági ellenőrzés a pontok számára (.shape[0] Tensor esetén)
        num_points = pcd_down.point["positions"].shape[0]
        if num_points < 3:
            self.get_logger().warn(f'Nincs elég pont a mintavételezés után: {num_points}')
            return

        # 4. Talajszűrés RANSAC segítségével (Tensor API)
        # Megjegyzés: A t.geometry-nél a segment_plane egy maszkot (Bool) és a síkmodellt adja vissza
        plane_model, inliers_mask = pcd_down.segment_plane(
            distance_threshold=self.ground_dist,
            ransac_n=3,
            num_iterations=100
        )
        
        # Nem-talaj pontok kiválasztása a maszk invertálásával (.logical_not())
        # Így tiszta Bool maszkot kapunk, ami nem dob hibát a select_by_mask-nál
        outliers_mask = inliers_mask.logical_not()
        pcd_no_ground = pcd_down.select_by_index(inliers_mask, invert=True)  # Nem-talaj pontok kiválasztása
        
        if pcd_no_ground.point["positions"].shape[0] == 0:
            return

        # 5. Klaszterezés (DBSCAN)
        # A Tensor-alapú cluster_dbscan jelenleg CPU-ra kényszerítheti az adatot az algoritmus jellege miatt,
        # de a t.geometry-n hívva egy Int32-es Open3D Tensort ad vissza.
        labels_tensor = pcd_no_ground.cluster_dbscan(eps=self.cluster_tol, min_points=self.min_cluster)
        labels = labels_tensor.cpu().numpy() # Átváltjuk numpy-ra a feldolgozáshoz

        # Detection3DArray előkészítése
        detections = Detection3DArray()
        detections.header = msg.header
        
        max_label = labels.max() if labels.size > 0 else -1
        
        # A pozíciókat visszahozzuk CPU-ra a bounding box számításokhoz
        pcd_no_ground_cpu = pcd_no_ground.cpu().to_legacy()

        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) < self.min_cluster:
                continue
                
            cluster_pcd = pcd_no_ground_cpu.select_by_index(cluster_indices)
            
            # Bounding Box számítás (Legacy API-n stabilabb a ROS üzenetekhez)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()
            
            detection = Detection3D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            hp = ObjectHypothesis()
            hp.class_id = f'cluster_{i}'
            hp.score = 1.0
            hypothesis.hypothesis = hp
            
            hypothesis.pose = PoseWithCovariance()
            hypothesis.pose.pose = Pose()
            hypothesis.pose.pose.position.x = center[0]
            hypothesis.pose.pose.position.y = center[1]
            hypothesis.pose.pose.position.z = center[2]
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            
            bbox = BoundingBox3D()
            bbox.center.position.x = center[0]
            bbox.center.position.y = center[1]
            bbox.center.position.z = center[2]
            bbox.size.x = extent[0]
            bbox.size.y = extent[1]
            bbox.size.z = extent[2]
            
            detection.bbox = bbox
            detections.detections.append(detection)

        # Publikálás
        self.cluster_pub.publish(detections)
        
        # Hibakeresési felhő visszaküldése ROS-ba
        no_ground_points = pcd_no_ground.point["positions"].cpu().numpy()
        no_ground_msg = self.numpy_to_pointcloud2(no_ground_points, msg.header)
        self.no_ground_pub.publish(no_ground_msg)

    def pointcloud2_to_numpy(self, msg):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        structured_array = np.fromiter(gen, dtype=dtype)
        if structured_array.size > 0:
            points = structured_array.view(np.float32).reshape(-1, 3)
            return points[np.isfinite(points).all(axis=1)]
        return np.zeros((0, 3), dtype=np.float32)

    def numpy_to_pointcloud2(self, points, header):
        return point_cloud2.create_cloud_xyz32(header, points)

def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
