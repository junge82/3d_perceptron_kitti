import os
import glob
import datetime
import numpy as np
import cv2
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField, NavSatFix, CameraInfo
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import serialize_message
import scipy.spatial.transform

def parse_kitti_timestamps(timestamp_path):
    """Parses standard KITTI human-readable timestamps into integer nanoseconds."""
    timestamps = []
    if not os.path.exists(timestamp_path):
        return None
    with open(timestamp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dt_str, ns_str = line.split('.')
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            total_ns = int(dt.timestamp() * 1e9) + int(ns_str[:9].ljust(9, '0'))
            timestamps.append(total_ns)
    return timestamps

def create_fields():
    return [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]

def convert_bin_to_pointcloud2(bin_path, stamp):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = 'velodyne'
    msg.height = 1
    msg.width = len(scan)
    msg.fields = create_fields()
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = scan.tobytes()
    return msg

def parse_oxts_to_messages(oxts_path, stamp):
    """Parses a single raw OXTS tracking line to native ROS Geo/Nav items."""
    with open(oxts_path, 'r') as f:
        data = [float(x) for x in f.read().split()]
    if not data:
        return None, None
        
    fix = NavSatFix()
    fix.header.stamp = stamp
    fix.header.frame_id = 'gps'
    fix.latitude = data[0]
    fix.longitude = data[1]
    fix.altitude = data[2]
    
    twist = TwistStamped()
    twist.header.stamp = stamp
    twist.header.frame_id = 'gps'
    twist.twist.linear.x = data[8]   # East speed
    twist.twist.linear.y = data[9]   # North speed
    twist.twist.linear.z = data[10]  # Upward speed
    
    return fix, twist

def parse_velo_to_cam_calib(calib_path, stamp):
    """Parses calib_velo_to_cam.txt and builds a ROS2 TransformStamped message."""
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file not found at {calib_path}")
        return None
        
    calib_data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip() or ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            if key in ['R', 'T']:
                calib_data[key] = np.array([float(x) for x in val.split()])

    if 'R' not in calib_data or 'T' not in calib_data:
        return None

    R = calib_data['R'].reshape(3, 3)
    T = calib_data['T']

    r = scipy.spatial.transform.Rotation.from_matrix(R)
    quat = r.as_quat()

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = 'camera_ref'
    t.child_frame_id = 'velodyne'
    
    t.transform.translation.x = T[0]
    t.transform.translation.y = T[1]
    t.transform.translation.z = T[2]
    
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    
    return t

def parse_cam_to_cam_calib(calib_path):
    """Parses calib_cam_to_cam.txt matrices safely for camera 02 and 03."""
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file not found at {calib_path}")
        return None, None

    calib_data = {}
    # Target prefixes to extract numbers from
    target_prefixes = ('S_rect_', 'K_', 'D_', 'R_rect_', 'P_rect_')
    
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip() or ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            
            # FIX: Only extract relevant numeric matrices, ignore dates/text metadata
            if key.startswith(target_prefixes):
                calib_data[key] = np.array([float(x) for x in val.split()])

    def build_info_msg(cam_idx, frame_id):
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        
        # Dimensions (S_rect contains 2 elements: [width, height])
        size_key = f"S_rect_{cam_idx}"
        if size_key in calib_data and len(calib_data[size_key]) >= 2:
            msg.width = int(calib_data[size_key][0])
            msg.height = int(calib_data[size_key][1])
            
        # Distortion Model (KITTI raw data is already rectified, so D elements are 0)
        msg.distortion_model = "plumb_bob"
        msg.d = list(calib_data.get(f"D_{cam_idx}", np.zeros(5)))
        
        # Intrinsics (3x3 Matrix flattened to 9 elements)
        msg.k = list(calib_data.get(f"K_{cam_idx}", np.eye(3).flatten()))
        
        # Rectification (3x3 Matrix flattened to 9 elements)
        msg.r = list(calib_data.get(f"R_rect_{cam_idx}", np.eye(3).flatten()))
        
        # Projection (3x4 Matrix flattened to 12 elements)
        msg.p = list(calib_data.get(f"P_rect_{cam_idx}", np.zeros(12)))
        
        return msg

    cam_left_info = build_info_msg("02", "camera_left")
    cam_right_info = build_info_msg("03", "camera_right")
    return cam_left_info, cam_right_info

def create_kitti_raw_bag(sequence_dir, calib_dir, bag_path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'), 
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', 
            output_serialization_format='cdr'
        )
    )
    
    # Structure topic dictionary including CameraInfo streams
    topics = {
        '/kitti/camera_left/image_raw': 'sensor_msgs/msg/Image',
        '/kitti/camera_left/camera_info': 'sensor_msgs/msg/CameraInfo',
        '/kitti/camera_right/image_raw': 'sensor_msgs/msg/Image',
        '/kitti/camera_right/camera_info': 'sensor_msgs/msg/CameraInfo',
        '/kitti/velodyne/pointcloud': 'sensor_msgs/msg/PointCloud2',
        '/kitti/oxts/fix': 'sensor_msgs/msg/NavSatFix',
        '/kitti/oxts/twist': 'geometry_msgs/msg/TwistStamped',
        '/tf_static': 'tf2_msgs/msg/TFMessage'
    }
    
    for idx, (name, msg_type) in enumerate(topics.items()):
        topic_metadata = rosbag2_py.TopicMetadata(idx, name, msg_type, 'cdr')
        writer.create_topic(topic_metadata)
        
    bridge = CvBridge()
    
    # Subdirectory structural mappings
    left_dir = os.path.join(sequence_dir, 'image_02', 'data')
    right_dir = os.path.join(sequence_dir, 'image_03', 'data')
    lidar_dir = os.path.join(sequence_dir, 'velodyne_points', 'data')
    oxts_dir = os.path.join(sequence_dir, 'oxts', 'data')
    
    timestamps = parse_kitti_timestamps(os.path.join(sequence_dir, 'velodyne_points', 'timestamps.txt'))
    
    left_imgs = sorted(glob.glob(os.path.join(left_dir, '*.png')))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, '*.png')))
    lidar_bins = sorted(glob.glob(os.path.join(lidar_dir, '*.bin')))
    oxts_files = sorted(glob.glob(os.path.join(oxts_dir, '*.txt')))
    
    total_frames = min(len(left_imgs), len(right_imgs), len(lidar_bins), len(oxts_files), len(timestamps))
    print(f"Generating true-timed continuous bag from {total_frames} raw frames...")

    # 1. Parse Global Static Transforms & Static Info
    cam_info_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    base_left_info, base_right_info = parse_cam_to_cam_calib(cam_info_path)

    if total_frames > 0:
        initial_time_ns = timestamps[0]
        initial_stamp = Time(nanoseconds=initial_time_ns).to_msg()
        
        calib_file_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
        static_transform = parse_velo_to_cam_calib(calib_file_path, initial_stamp)
        
        if static_transform is not None:
            tf_static_msg = TFMessage()
            tf_static_msg.transforms.append(static_transform)
            writer.write('/tf_static', serialize_message(tf_static_msg), initial_time_ns)
            print("Successfully recorded calibration static transform matrix to /tf_static.")

    # 2. Data generation timeline loop
    for i in range(total_frames):
        current_time_ns = timestamps[i]
        stamp = Time(nanoseconds=current_time_ns).to_msg()
        
        # 1. Left Color Cam + Info
        cv_left = cv2.imread(left_imgs[i], cv2.IMREAD_COLOR)
        if cv_left is not None:
            msg_left = bridge.cv2_to_imgmsg(cv_left, encoding="bgr8")
            msg_left.header.stamp = stamp
            msg_left.header.frame_id = 'camera_left'
            writer.write('/kitti/camera_left/image_raw', serialize_message(msg_left), current_time_ns)
            
            if base_left_info is not None:
                base_left_info.header.stamp = stamp
                writer.write('/kitti/camera_left/camera_info', serialize_message(base_left_info), current_time_ns)
            
        # 2. Right Color Cam + Info
        cv_right = cv2.imread(right_imgs[i], cv2.IMREAD_COLOR)
        if cv_right is not None:
            msg_right = bridge.cv2_to_imgmsg(cv_right, encoding="bgr8")
            msg_right.header.stamp = stamp
            msg_right.header.frame_id = 'camera_right'
            writer.write('/kitti/camera_right/image_raw', serialize_message(msg_right), current_time_ns)
            
            if base_right_info is not None:
                base_right_info.header.stamp = stamp
                writer.write('/kitti/camera_right/camera_info', serialize_message(base_right_info), current_time_ns)

            
        # 3. LiDAR Scans
        msg_lidar = convert_bin_to_pointcloud2(lidar_bins[i], stamp)
        writer.write('/kitti/velodyne/pointcloud', serialize_message(msg_lidar), current_time_ns)
        
        # 4. GPS Fix & Twist State Matrix
        msg_fix, msg_twist = parse_oxts_to_messages(oxts_files[i], stamp)
        if msg_fix is not None:
            writer.write('/kitti/oxts/fix', serialize_message(msg_fix), current_time_ns)
            writer.write('/kitti/oxts/twist', serialize_message(msg_twist), current_time_ns)
            
        if i % 25 == 0 or i == total_frames - 1:
            print(f"Pack progress status: Frame {i+1}/{total_frames}")

    del writer
    print(f"\nSuccess! Continuous raw sequence saved to: {bag_path}")

if __name__ == '__main__':
    # Update paths to point directly to your dataset structure
    KITTI_SEQUENCE = "/mnt/fastDisk/kitti3d/raw_data_downloader/2011_09_26/2011_09_26_drive_0029_sync"
    KITTI_CALIB_DIR = "/mnt/fastDisk/kitti3d/raw_data_downloader/2011_09_26" # Dir holding calib_velo_to_cam.txt
    BAG_OUTPUT = "2011_09_26_drive_0029_sync"
    
    rclpy.init()
    create_kitti_raw_bag(KITTI_SEQUENCE, KITTI_CALIB_DIR, BAG_OUTPUT)
    rclpy.shutdown()
