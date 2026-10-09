import os
import unittest
from pathlib import Path

try:
    import rosbag2_py
    import yaml
    from kitti_to_mcap import SENSOR_FILE_PATTERNS, TOPICS, parse_kitti_timestamps
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as error:
    MISSING_DEPENDENCIES = str(error)
else:
    MISSING_DEPENDENCIES = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
BAG_PATH = Path(
    os.environ.get(
        'KITTI_MCAP_BAG',
        str(PROJECT_DIR / '2011_09_29_drive_0071_sync'),
    )
)
SEQUENCE_PATH = Path(
    os.environ.get(
        'KITTI_SEQUENCE',
        '/mnt/fastDisk/kitti3d/raw_data_downloader/2011_09_29/'
        '2011_09_29_drive_0071_sync',
    )
)


@unittest.skipIf(MISSING_DEPENDENCIES, f'ROS MCAP test dependencies unavailable: {MISSING_DEPENDENCIES}')
@unittest.skipUnless((BAG_PATH / 'metadata.yaml').is_file(), f'MCAP bag not found: {BAG_PATH}')
class GeneratedMcapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (BAG_PATH / 'metadata.yaml').open() as metadata_file:
            cls.metadata = yaml.safe_load(metadata_file)['rosbag2_bagfile_information']
        cls.topic_counts = {
            topic['topic_metadata']['name']: topic['message_count']
            for topic in cls.metadata['topics_with_message_count']
        }

    def test_topics_and_counts_match_source_streams(self):
        self.assertEqual(self.metadata['storage_identifier'], 'mcap')
        self.assertEqual(
            {
                topic['topic_metadata']['name']: topic['topic_metadata']['type']
                for topic in self.metadata['topics_with_message_count']
            },
            TOPICS,
        )

        if not SEQUENCE_PATH.is_dir():
            self.skipTest(f'KITTI source sequence not found: {SEQUENCE_PATH}')

        source_counts = {}
        for sensor_name, pattern in SENSOR_FILE_PATTERNS.items():
            sensor_dir = SEQUENCE_PATH / sensor_name
            files = sorted((sensor_dir / 'data').glob(pattern))
            timestamps = parse_kitti_timestamps(str(sensor_dir / 'timestamps.txt'))
            self.assertEqual(len(files), len(timestamps), sensor_name)
            source_counts[sensor_name] = len(files)

        expected_counts = {
            '/kitti/camera_left/image_raw': source_counts['image_02'],
            '/kitti/camera_left/camera_info': source_counts['image_02'],
            '/kitti/camera_right/image_raw': source_counts['image_03'],
            '/kitti/camera_right/camera_info': source_counts['image_03'],
            '/kitti/velodyne/pointcloud': source_counts['velodyne_points'],
            '/kitti/oxts/fix': source_counts['oxts'],
            '/kitti/oxts/twist': source_counts['oxts'],
            '/tf_static': int(source_counts['velodyne_points'] > 0),
        }
        self.assertEqual(self.topic_counts, expected_counts)

    def test_sample_headers_match_sensor_timestamps(self):
        if not SEQUENCE_PATH.is_dir():
            self.skipTest(f'KITTI source sequence not found: {SEQUENCE_PATH}')
        first_timestamps = {
            sensor_name: parse_kitti_timestamps(
                str(SEQUENCE_PATH / sensor_name / 'timestamps.txt')
            )[0]
            for sensor_name in SENSOR_FILE_PATTERNS
        }
        expected_timestamps = {
            '/kitti/camera_left/image_raw': first_timestamps['image_02'],
            '/kitti/camera_left/camera_info': first_timestamps['image_02'],
            '/kitti/camera_right/image_raw': first_timestamps['image_03'],
            '/kitti/camera_right/camera_info': first_timestamps['image_03'],
            '/kitti/velodyne/pointcloud': first_timestamps['velodyne_points'],
            '/kitti/oxts/fix': first_timestamps['oxts'],
            '/kitti/oxts/twist': first_timestamps['oxts'],
            '/tf_static': first_timestamps['velodyne_points'],
        }

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(BAG_PATH), storage_id='mcap'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr',
            ),
        )
        message_types = {
            topic.name: get_message(topic.type)
            for topic in reader.get_all_topics_and_types()
        }
        remaining_topics = set(expected_timestamps)

        while reader.has_next() and remaining_topics:
            topic_name, serialized, record_timestamp = reader.read_next()
            if topic_name not in remaining_topics:
                continue

            message = deserialize_message(serialized, message_types[topic_name])
            if topic_name == '/tf_static':
                header_timestamps = [
                    transform.header.stamp.sec * 1_000_000_000
                    + transform.header.stamp.nanosec
                    for transform in message.transforms
                ]
                self.assertTrue(header_timestamps)
                self.assertEqual(header_timestamps, [record_timestamp] * len(header_timestamps))
            else:
                stamp = message.header.stamp
                header_timestamp = stamp.sec * 1_000_000_000 + stamp.nanosec
                self.assertEqual(header_timestamp, record_timestamp)

            self.assertEqual(record_timestamp, expected_timestamps[topic_name], topic_name)
            remaining_topics.remove(topic_name)

        self.assertFalse(remaining_topics, f'No sample read for: {sorted(remaining_topics)}')


if __name__ == '__main__':
    unittest.main()