import os
import glob
import datetime
import numpy as np
import cv2
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField, NavSatFix
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import serialize_message

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
            # KITTI raw timestamp layout example: 2011-09-26 13:02:25.123456789
            dt_str, ns_str = line.split('.')
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            # Combine unix timestamp seconds with the fractional nanoseconds chunk
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
        
    # Lat/Lon/Alt indices
    fix = NavSatFix()
    fix.header.stamp = stamp
    fix.header.frame_id = 'gps'
    fix.latitude = data[0]
    fix.longitude = data[1]
    fix.altitude = data[2]
    
    # Linear velocities (V_east, V_north) converted to Twist message
    twist = TwistStamped()
    twist.header.stamp = stamp
    twist.header.frame_id = 'gps'
    twist.twist.linear.x = data[8]   # East speed
    twist.twist.linear.y = data[9]   # North speed
    twist.twist.linear.z = data[10]  # Upward speed
    
    return fix, twist

def create_kitti_raw_bag(sequence_dir, bag_path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    
    # 5 Stream topics to structure camera, tracking and vehicle velocity
    topics = {
        '/kitti/camera_left/image_raw': 'sensor_msgs/msg/Image',
        '/kitti/camera_right/image_raw': 'sensor_msgs/msg/Image',
        '/kitti/velodyne/pointcloud': 'sensor_msgs/msg/PointCloud2',
        '/kitti/oxts/fix': 'sensor_msgs/msg/NavSatFix',
        '/kitti/oxts/twist': 'geometry_msgs/msg/TwistStamped'
    }
    
    for name, msg_type in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(0, name, msg_type, 'cdr'))
        
    bridge = CvBridge()
    
    # Setup subdirectory mappings
    left_dir = os.path.join(sequence_dir, 'image_02', 'data')
    right_dir = os.path.join(sequence_dir, 'image_03', 'data')
    lidar_dir = os.path.join(sequence_dir, 'velodyne_points', 'data')
    oxts_dir = os.path.join(sequence_dir, 'oxts', 'data')
    
    # Parse master synchronized timing vectors
    timestamps = parse_kitti_timestamps(os.path.join(sequence_dir, 'velodyne_points', 'timestamps.txt'))
    
    left_imgs = sorted(glob.glob(os.path.join(left_dir, '*.png')))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, '*.png')))
    lidar_bins = sorted(glob.glob(os.path.join(lidar_dir, '*.bin')))
    oxts_files = sorted(glob.glob(os.path.join(oxts_dir, '*.txt')))
    
    total_frames = min(len(left_imgs), len(right_imgs), len(lidar_bins), len(oxts_files), len(timestamps))
    print(f"Generating true-timed continuous bag from {total_frames} raw frames...")

    for i in range(total_frames):
        current_time_ns = timestamps[i]
        stamp = Time(nanoseconds=current_time_ns).to_msg()
        
        # 1. Left Color Cam
        cv_left = cv2.imread(left_imgs[i])
        if cv_left is not None:
            msg_left = bridge.cv2_to_imgmsg(cv_left, encoding="bgr8")
            msg_left.header.stamp = stamp
            msg_left.header.frame_id = 'camera_left'
            writer.write('/kitti/camera_left/image_raw', serialize_message(msg_left), current_time_ns)
            
        # 2. Right Color Cam
        cv_right = cv2.imread(right_imgs[i])
        if cv_right is not None:
            msg_right = bridge.cv2_to_imgmsg(cv_right, encoding="bgr8")
            msg_right.header.stamp = stamp
            msg_right.header.frame_id = 'camera_right'
            writer.write('/kitti/camera_right/image_raw', serialize_message(msg_right), current_time_ns)
            
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

    print(f"\nSuccess! Continuous raw sequence saved to: {bag_path}")

if __name__ == '__main__':
    # Make sure to point this path directly to the extracted inner folder 
    KITTI_SEQUENCE = "/mnt/fastDisk/kitti3d/raw_data_downloader/2011_09_29/2011_09_29_drive_0071_sync"
    BAG_OUTPUT = "2011_09_29_drive_0071_sync"
    
    rclpy.init()
    create_kitti_raw_bag(KITTI_SEQUENCE, BAG_OUTPUT)
    rclpy.shutdown()
