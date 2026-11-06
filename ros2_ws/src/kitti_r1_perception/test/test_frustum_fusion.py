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

from types import SimpleNamespace

from kitti_r1_perception.frustum_fusion_node import FrustumFusionNode
import numpy as np
from std_msgs.msg import Header
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

    camera_intrinsics = np.array([
        [721.5377, 0.0, 609.5593],
        [0.0, 721.5377, 172.854],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    lidar_to_camera = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    node.K_tensor = torch.from_numpy(camera_intrinsics)
    node.R_tensor = torch.from_numpy(lidar_to_camera[:3, :3])
    node.t_tensor = torch.from_numpy(lidar_to_camera[:3, 3])
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
