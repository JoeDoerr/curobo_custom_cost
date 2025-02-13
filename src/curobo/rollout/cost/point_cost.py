from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo.geom.transform import batch_transform_points, transform_points, pose_to_matrix
from curobo.curobolib.geom import PoseError, PoseErrorDistance
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import OrientationError, Pose
from curobo.util.logger import log_error

# Local Folder
from .cost_base import CostBase, CostConfig

import torch.nn.functional as F
import rospy
import time
import torch
"""
Take in the set of non-colliding rays
Move towards the closet n rays to the given position or choose one out a range of closest randomly so that we don't move to an averaged position
Heavily front-weight this as once we get to the closet ray it will no longer be a ray (do high start to negative end, and then do all negative turn to 0 for the scaling)
"""


class PointCost(CostBase):

    def __init__(self, config: CostConfig = None):
        device = torch.device("cuda:0")
        self.weight = torch.tensor(1.0, device=device)
        self.tensor_args = TensorDeviceType()
        zd = 0.1
        xd = 0.2
        self.points = torch.tensor(
            [
                [0., 0., 0.],
                [xd, 0., zd],
                [-xd, 0, zd],
            ], device=device
        )
        self.target_position = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.target_quaternion = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device
        )
        CostBase.__init__(self, config)

    def forward(self, gripper_pos_batch, gripper_rot_batch):
        '''
        input:
        pos: [batch, horizon, 3]
        rot: [batch, horizon, 4]
        '''
        batch = gripper_pos_batch.shape[0]
        horizon = gripper_pos_batch.shape[1]
        # print('input shape:', gripper_pos_batch.shape, gripper_rot_batch.shape)
        batch_gripper_points = self.points.unsqueeze(0).unsqueeze(0).repeat(
            gripper_pos_batch.shape[0], gripper_pos_batch.shape[1], 1, 1
        )  # [batch, horizon, 3, 3]
        # print("batch_gripper_points", batch_gripper_points.shape)
        # matrix = pose_to_matrix(
        #     gripper_pos_batch,
        #     gripper_rot_batch,
        # )
        # print("matrix", matrix)
        # print("matrix", matrix.shape)
        transformed_points = batch_transform_points(
            gripper_pos_batch.reshape(-1, 3),
            gripper_rot_batch.reshape(-1, 4),
            batch_gripper_points.reshape(-1, 3, 3),
        )  # [batch * horizon, 3]
        # print("transformed_points", transformed_points.shape)
        transformed_points = transformed_points.reshape(batch, horizon, 3, 3)
        # print("transformed_points", transformed_points.shape)
        target_points = transform_points(
            self.target_position,
            self.target_quaternion,
            self.points,
        )  # [3, 3]
        target_points = target_points.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        # print("target_points", target_points.shape)

        distances = torch.linalg.norm(
            transformed_points - target_points, dim=-1
        )
        # print("distances", distances)

        cost = torch.sum(distances, dim=-1)
        # cost = torch.exp(100 * cost) - 1
        cost *= 10000

        # take minimum of symmetric cost; maybe not

        return cost * self.weight
