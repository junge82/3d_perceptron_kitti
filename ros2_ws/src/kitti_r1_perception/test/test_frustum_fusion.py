# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the frustum-fusion callback."""

import pytest
from types import SimpleNamespace

from kitti_r1_perception.calibration import (
    K_KITTI,
    T_LIDAR_TO_CAM,
)
from kitti_r1_perception.frustum_fusion_node import FrustumFusionNode

from std_msgs.msg import Header
import numpy as np
import torch
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FusionHarness(FrustumFusionNode):
    def get_parameter(self, _name):
        return SimpleNamespace(value=self.match_radius)

    def get_logger(self):
        return self.logger


def _make_fusion_harness(match_radius=50.0):
    node = object.__new__(_FusionHarness)
    node.device = torch.device('cpu')
    node.is_cuda = False
    node.match_radius = match_radius
    node.logger = _Logger()
    node.fused_pub = _Publisher()
    node.K_tensor = torch.from_numpy(K_KITTI)
    node.R_tensor = torch.from_numpy(T_LIDAR_TO_CAM[:3, :3])
    node.t_tensor = torch.from_numpy(T_LIDAR_TO_CAM[:3, 3])
    return node


def _make_messages(camera_center):
    header = Header()
    header.stamp.sec = 1

    lidar_detection = Detection3D()
    lidar_detection.bbox.center.position.x = 10.0
    lidar_detection.bbox.center.position.y = 0.0
    lidar_detection.bbox.center.position.z = 0.0
    lidar_detection.bbox.size.x = 4.0
    lidar_detection.bbox.size.y = 1.8
    lidar_detection.bbox.size.z = 1.5

    lidar_message = Detection3DArray()
    lidar_message.header = header
    lidar_message.detections.append(lidar_detection)

    camera_detection = Detection2D()
    camera_detection.bbox.center.position.x = camera_center[0]
    camera_detection.bbox.center.position.y = camera_center[1]
    camera_result = ObjectHypothesisWithPose()
    camera_result.hypothesis.class_id = '0'
    camera_result.hypothesis.score = 0.9
    camera_detection.results.append(camera_result)

    camera_message = Detection2DArray()
    camera_message.header = header
    camera_message.detections.append(camera_detection)
    return camera_message, lidar_message


def test_sync_callback_projects_and_publishes_matched_detection():
    """Publish a detection when the projected LiDAR point matches a camera box."""
    node = _make_fusion_harness(match_radius=10.0)
    camera_message, lidar_message = _make_messages((615.0, 178.0))

    node.sync_callback(camera_message, lidar_message)

    assert len(node.fused_pub.messages) == 1
    fused_message = node.fused_pub.messages[0]
    assert len(fused_message.detections) == 1
    result = fused_message.detections[0].results[0].hypothesis
    assert result.class_id == '0'
    assert result.score == 0.9

def test_sync_callback_does_not_publish_unmatched_detection():
    """Do not publish when the projected LiDAR point is outside the match radius."""
    node = _make_fusion_harness(match_radius=10.0)
    camera_message, lidar_message = _make_messages((1000.0, 300.0))

    node.sync_callback(camera_message, lidar_message)

    assert node.fused_pub.messages == []

def test_extract_centers_valid_2d_and_3d():
    """Verify 2D and 3D center extraction returns properly shaped float32 arrays."""
    node = _make_fusion_harness()
    camera_msg, lidar_msg = _make_messages((615.0, 178.0))

    cam_pts = node.extract_centers(camera_msg, is_3d=False)
    lidar_pts = node.extract_centers(lidar_msg, is_3d=True)

    assert cam_pts.shape == (1, 2)
    assert cam_pts.dtype == np.float32
    np.testing.assert_allclose(cam_pts[0], [615.0, 178.0])

    assert lidar_pts.shape == (1, 3)
    assert lidar_pts.dtype == np.float32
    np.testing.assert_allclose(lidar_pts[0], [10.0, 0.0, 0.0])

def test_extract_centers_empty_detections():
    """Verify empty detection arrays return (0, N) shaped arrays without crashing."""
    empty_cam = Detection2DArray()
    empty_lidar = Detection3DArray()

    node = _make_fusion_harness()
    assert node.extract_centers(empty_cam, is_3d=False).shape == (0, 2)
    assert node.extract_centers(empty_lidar, is_3d=True).shape == (0, 3)

def test_extract_centers_raises_on_missing_z():
    """Verify ValueError is raised when 3D extraction is requested on 2D detections."""
    camera_msg, _ = _make_messages((615.0, 178.0))
    node = _make_fusion_harness()

    with pytest.raises(ValueError, match="'z' attribute is missing"):
        node.extract_centers(camera_msg, is_3d=True)