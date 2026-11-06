import os
import glob
import datetime
import heapq
import shutil
import numpy as np
import cv2
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField, NavSatFix, CameraInfo
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import serialize_message
import scipy.spatial.transform

TOPICS = {
    '/kitti/camera_left/image_raw': 'sensor_msgs/msg/Image',
    '/kitti/camera_left/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/kitti/camera_right/image_raw': 'sensor_msgs/msg/Image',
    '/kitti/camera_right/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/kitti/velodyne/pointcloud': 'sensor_msgs/msg/PointCloud2',
    '/kitti/oxts/fix': 'sensor_msgs/msg/NavSatFix',
    '/kitti/oxts/twist': 'geometry_msgs/msg/TwistStamped',
    '/tf_static': 'tf2_msgs/msg/TFMessage',
}

SENSOR_FILE_PATTERNS = {
    'image_02': '*.png',
    'image_03': '*.png',
    'velodyne_points': '*.bin',
    'oxts': '*.txt',
}

CAMERA_STREAMS = {
    'image_02': (
        'camera_left',
        '/kitti/camera_left/image_raw',
        '/kitti/camera_left/camera_info',
    ),
    'image_03': (
        'camera_right',
        '/kitti/camera_right/image_raw',
        '/kitti/camera_right/camera_info',
    ),
}


def report_progress(message):
    print(f"[mcap] {message}", flush=True)


def parse_kitti_timestamps(timestamp_path):
    """Parses standard KITTI human-readable timestamps into integer nanoseconds."""
    timestamps = []
    if not os.path.exists(timestamp_path):
        raise FileNotFoundError(f"KITTI timestamp file not found: {timestamp_path}")
    with open(timestamp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dt_str, ns_str = line.rsplit('.', 1)
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            delta = dt - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            seconds = delta.days * 86400 + delta.seconds
            # Store UTC time and its fractional part as integer nanoseconds.
            total_ns = seconds * 1_000_000_000 + int(ns_str[:9].ljust(9, '0'))
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
    # KITTI stores each point as four float32 values: x, y, z, intensity.
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = 'velodyne'
    msg.height = 1
    msg.width = len(scan)
    msg.fields = create_fields()
    msg.is_bigendian = False
    # Four float32 fields occupy 16 bytes per point.
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
    if len(data) < 11:
        raise ValueError(f"OXTS record must contain at least 11 values: {oxts_path}")
        
    fix = NavSatFix()
    fix.header.stamp = stamp
    fix.header.frame_id = 'gps'
    fix.latitude = data[0]
    fix.longitude = data[1]
    fix.altitude = data[2]
    
    twist = TwistStamped()
    twist.header.stamp = stamp
    twist.header.frame_id = 'base_link'
    twist.twist.linear.x = data[8]   # Forward speed
    twist.twist.linear.y = data[9]   # Leftward speed
    twist.twist.linear.z = data[10]  # Upward speed
    
    return fix, twist

def parse_velo_to_cam_calib(calib_path, cam_calib_data, stamp):
    """Builds static transforms from Velodyne to the rectified camera frames."""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
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
        raise ValueError(f"Calibration file must contain R and T: {calib_path}")

    velo_rotation = calib_data['R'].reshape(3, 3)
    velo_translation = calib_data['T']
    rectification = cam_calib_data['R_rect_00'].reshape(3, 3)
    camera_from_velodyne = {}

    for cam_idx in ('02', '03'):
        projection = cam_calib_data[f'P_rect_{cam_idx}'].reshape(3, 4)
        # Remove intrinsics from P's translation column to recover camera translation.
        camera_offset = np.linalg.solve(projection[:, :3], projection[:, 3])
        transform = np.eye(4)
        # Compose rectification and Velodyne extrinsics into camera-from-Velodyne.
        transform[:3, :3] = rectification @ velo_rotation
        transform[:3, 3] = rectification @ velo_translation + camera_offset
        camera_from_velodyne[cam_idx] = transform

    # Convert the two Velodyne-relative transforms into right-to-left.
    left_from_right = camera_from_velodyne['02'] @ np.linalg.inv(camera_from_velodyne['03'])
    transforms = []
    for parent_frame, child_frame, transform in (
        ('camera_left', 'velodyne', camera_from_velodyne['02']),
        ('camera_left', 'camera_right', left_from_right),
    ):
        # ROS stores each rigid transform's rotation as a quaternion.
        quat = scipy.spatial.transform.Rotation.from_matrix(transform[:3, :3]).as_quat()
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = transform[0, 3]
        msg.transform.translation.y = transform[1, 3]
        msg.transform.translation.z = transform[2, 3]
        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]
        transforms.append(msg)

    return transforms

def parse_cam_to_cam_calib(calib_path):
    """Parses calib_cam_to_cam.txt matrices safely for camera 02 and 03."""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calib_data = {}
    # Target prefixes to extract numbers from
    target_prefixes = ('S_rect_', 'R_rect_', 'P_rect_')
    
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
        projection_key = f"P_rect_{cam_idx}"
        size_key = f"S_rect_{cam_idx}"
        if projection_key not in calib_data or size_key not in calib_data:
            raise ValueError(f"Missing rectified camera calibration for camera {cam_idx}")
        projection = calib_data[projection_key].reshape(3, 4)
        
        # Dimensions (S_rect contains 2 elements: [width, height])
        msg.width = int(calib_data[size_key][0])
        msg.height = int(calib_data[size_key][1])
            
        # The image streams are rectified, so camera info uses zero distortion.
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        
        # Rectified intrinsics are the left 3x3 part of the projection matrix.
        msg.k = projection[:, :3].flatten().tolist()
        
        msg.r = np.eye(3).flatten().tolist()
        
        msg.p = projection.flatten().tolist()
        
        return msg

    cam_left_info = build_info_msg("02", "camera_left")
    cam_right_info = build_info_msg("03", "camera_right")
    return cam_left_info, cam_right_info, calib_data

def load_sensor_samples(sequence_dir):
    sensor_samples = {}
    for sensor_name, pattern in SENSOR_FILE_PATTERNS.items():
        report_progress(f"Scanning {sensor_name} files and timestamps...")
        sensor_dir = os.path.join(sequence_dir, sensor_name)
        data_files = sorted(
            glob.glob(os.path.join(sensor_dir, 'data', pattern))
        )
        timestamps = parse_kitti_timestamps(os.path.join(sensor_dir, 'timestamps.txt'))
        if len(data_files) != len(timestamps):
            raise ValueError(
                f"{sensor_name}: found {len(data_files)} data files and "
                f"{len(timestamps)} timestamps"
            )
        sensor_samples[sensor_name] = list(
            zip(timestamps, data_files, strict=True)
        )
        report_progress(f"{sensor_name}: {len(data_files)} samples ready")
    return sensor_samples


def _sensor_events(sensor_name, samples):
    for timestamp_ns, path in samples:
        yield timestamp_ns, sensor_name, path


def merge_sample_events(sensor_samples):
    streams = [
        _sensor_events(sensor_name, samples)
        for sensor_name, samples in sensor_samples.items()
    ]
    return heapq.merge(*streams)


def write_message(writer, topic, message, timestamp_ns):
    message.header.stamp = Time(nanoseconds=timestamp_ns).to_msg()
    writer.write(topic, serialize_message(message), timestamp_ns)

def write_sensor_sample(writer, bridge, sensor_name, path, timestamp_ns, camera_info):
    stamp = Time(nanoseconds=timestamp_ns).to_msg()

    if sensor_name in CAMERA_STREAMS:
        frame_id, image_topic, info_topic = CAMERA_STREAMS[sensor_name]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            report_progress(f"Warning: could not read camera image: {path}")
            return
        message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        message.header.frame_id = frame_id
        write_message(writer, image_topic, message, timestamp_ns)
        info_message = camera_info[frame_id]
        if info_message is not None:
            write_message(writer, info_topic, info_message, timestamp_ns)
    elif sensor_name == 'velodyne_points':
        message = convert_bin_to_pointcloud2(path, stamp)
        write_message(writer, '/kitti/velodyne/pointcloud', message, timestamp_ns)
    elif sensor_name == 'oxts':
        fix, twist = parse_oxts_to_messages(path, stamp)
        if fix is not None:
            write_message(writer, '/kitti/oxts/fix', fix, timestamp_ns)
            write_message(writer, '/kitti/oxts/twist', twist, timestamp_ns)


def remove_existing_bag(bag_path):
    output_path = os.path.abspath(bag_path)
    protected_paths = {os.path.abspath(os.sep), os.path.abspath(os.curdir)}
    if output_path in protected_paths:
        raise ValueError(f"Refusing to overwrite unsafe bag path: {output_path}")

    if os.path.islink(output_path) or os.path.isfile(output_path):
        report_progress(f"Removing existing output file: {output_path}")
        os.remove(output_path)
    elif os.path.isdir(output_path):
        report_progress(f"Removing existing bag directory: {output_path}")
        shutil.rmtree(output_path)


def create_kitti_raw_bag(sequence_dir, calib_dir, bag_path):
    report_progress("Starting KITTI to MCAP conversion")
    sensor_samples = load_sensor_samples(sequence_dir)
    report_progress("Loading camera calibration...")
    cam_info_path = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
    left_info, right_info, cam_calib_data = parse_cam_to_cam_calib(cam_info_path)
    camera_info = {
        'camera_left': left_info,
        'camera_right': right_info,
    }

    lidar_samples = sensor_samples['velodyne_points']
    static_tf = None
    if lidar_samples:
        report_progress("Preparing static camera-to-LiDAR transforms...")
        initial_time_ns = lidar_samples[0][0]
        stamp = Time(nanoseconds=initial_time_ns).to_msg()
        calibration_path = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
        transforms = parse_velo_to_cam_calib(
            calibration_path, cam_calib_data, stamp
        )
        static_tf = (TFMessage(transforms=transforms), initial_time_ns)

    bridge = CvBridge()
    report_progress(f"Preparing output bag: {bag_path}")
    remove_existing_bag(bag_path)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
    )
    for idx, (name, msg_type) in enumerate(TOPICS.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(idx, name, msg_type, 'cdr'))
    report_progress(f"MCAP ready with {len(TOPICS)} topics")

    if static_tf is not None:
        report_progress("Writing static transforms...")
        tf_message, timestamp_ns = static_tf
        writer.write('/tf_static', serialize_message(tf_message), timestamp_ns)

    total_samples = sum(len(samples) for samples in sensor_samples.values())
    progress_interval = max(1, total_samples // 50)
    report_progress(f"Writing {total_samples} sensor samples...")
    for index, (timestamp_ns, sensor_name, path) in enumerate(
        merge_sample_events(sensor_samples), start=1
    ):
        write_sensor_sample(writer, bridge, sensor_name, path, timestamp_ns, camera_info)
        if index % progress_interval == 0 or index == total_samples:
            report_progress(
                f"Processed {index}/{total_samples} samples "
                f"({index / total_samples:.0%})"
            )

    report_progress("Finalizing MCAP bag...")
    del writer
    report_progress(f"Conversion complete: {bag_path}")

if __name__ == '__main__':
    # Update paths to point directly to your dataset structure
    KITTI_SEQUENCE = "/Your-path/raw_data_downloader/2011_09_29/2011_09_29_drive_0071_sync"
    KITTI_CALIB_DIR = "/Your-path/raw_data_downloader/2011_09_26" # Dir holding calib_velo_to_cam.txt
    BAG_OUTPUT = "2011_09_29_drive_0071_sync"
    
    rclpy.init()
    create_kitti_raw_bag(KITTI_SEQUENCE, KITTI_CALIB_DIR, BAG_OUTPUT)
    rclpy.shutdown()
