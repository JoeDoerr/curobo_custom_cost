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
    
    def cost_window(self, gripper_pos_batch, gripper_rot_batch):
        '''
        input:
        pos: [batch, horizon, 3]
        rot: [batch, horizon, 4]
        '''
        batch = gripper_pos_batch.shape[0]
        horizon = gripper_pos_batch.shape[1]
        # print('input shape:', gripper_pos_batch.shape, gripper_rot_batch.shape)
        #There are points that are defined as relative to the reference point on the pose. We use fk to find their values for the current state and the desired state to calculate cost.
        batch_gripper_points = self.points.unsqueeze(0).unsqueeze(0).repeat(
            gripper_pos_batch.shape[0], gripper_pos_batch.shape[1], 1, 1
        )  # [batch, horizon, 3, 3]

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
        ) #[batch, horizon, 3]
        distances = torch.sum(distances, dim=-1) #[batch, horizon]

        #Find the first one on the horizon dimension. 
        #We need to use torch min so we need to do a mask on arange and turn on the values that are true
        valid = distances < 0.2 #threshold
        #print("distances.shape", distances.shape, valid.shape)
        #print("arange", torch.arange(horizon, device=torch.device("cuda:0")).unsqueeze(0).shape)
        counting_up = torch.arange(horizon, device=torch.device("cuda:0")).unsqueeze(0).repeat(batch, 1) #[batch, horizon]
        indexing_when_true = torch.where(valid, counting_up, torch.full_like(distances, 100000, device=torch.device("cuda:0"))) #torch.where(true/false mask, put in when true, put in when false)
        #print("indexing_when_true.shape", indexing_when_true.shape)
        #[USEFUL] Now indexing_when_true has the timestep of success at successful places and 1e5 at unsuccessful places
        earliest_converged = indexing_when_true.argmin(dim=-1) #[batch]
        earliest_converged[earliest_converged == 0] = (distances.shape[1] - 1) #No converges just put the last position as it
        #print("earliest_converged.shape", earliest_converged.shape)
        #[USEFUL] Earliest_converged is a [batch] size with index of soonest point on trajectory that is below the threshold distance
        #Now we need to scale distances by a scaling [batch, horizon] that is the position of earliest converged with steps going # backwards
        steps_scale = 10
        steps_scale = min(steps_scale, horizon)
        start_scaling = earliest_converged - steps_scale #Start the cost 8 steps back
        start_scaling[start_scaling < 0] = 0
        #print("start scaling.shape", start_scaling.shape)
        #[USEFUL] Now start_scaling is the position we start the scaling from

        scaling_values = torch.linspace(0.2, 1.0, steps_scale, device=torch.device("cuda:0")).unsqueeze(0).repeat(batch, 1) #[batch, steps_scale]
        batch_indices = torch.arange(batch).unsqueeze(1)  #[batch, 1]
        column_indices = start_scaling.unsqueeze(1) + torch.arange(steps_scale, dtype=torch.int, device=torch.device("cuda:0")).unsqueeze(0) #[batch, steps_scale]
        scaling_matrix = torch.full_like(distances, 0.0, device=torch.device("cuda:0"))
        scaling_matrix[batch_indices, column_indices] = scaling_values
        # print("distances", distances)
        cost = distances * scaling_matrix

        #cost = torch.sum(distances, dim=-1)
        cost *= 10000 #for mpc 10000
        print("cost.shape", cost.shape, cost.mean(), earliest_converged.float().mean())
        """
        The cost should go down the sooner we get there it. The function at a given step should be differentiable and vary smoothly across optimization steps. 
        The points leading up to getting there should have a will to improve it directed by gradients. 
        The cost signal is simply the points leading up to getting to the goal have a higher euclidean distance cost to the goal. So see if we can get there earlier. 
        The other costs that are after or before just have no cost. The multiplier on the eucldiean distance cost as a whole is scaled by how far in the trajectory it is. 
        The system differentiably will try to do better somewhat locally from where it succeeded, effectively attempting to get there faster while preserving the trajectory.
        We can't just say to get to it at each step and try to minimize the trajectory size as that will go local minima
        We would at least to have a moving last step. But if we just do one step then the other steps want to be shorter but don't have gradient guidance for shorter. 

        The idea is to not lose the solution from initialization and not do suboptimal movement just to be closer, but to incrementally see if we can get there faster.
        """

        # take minimum of symmetric cost; maybe not

        return cost * self.weight

    def distance_scaling(self, distances, start=10.0, end=1000.0):
        batch_size = distances.shape[0]
        horizon = distances.shape[1]
        step = (end - start) / horizon
        out = torch.arange(start, end, step, device=torch.device("cuda:0"))
        out = out.unsqueeze(0).repeat(batch_size, 1)
        out[out < 0.0] = 0.0
        return out

    def snake(self, gripper_pos_batch, gripper_rot_batch):
        #return self.cost_window(gripper_pos_batch, gripper_rot_batch)
        '''
        input:
        pos: [batch, horizon, 3]
        rot: [batch, horizon, 4]
        '''
        batch = gripper_pos_batch.shape[0]
        horizon = gripper_pos_batch.shape[1]
        # print('input shape:', gripper_pos_batch.shape, gripper_rot_batch.shape)
        #There are points that are defined as relative to the reference point on the pose. We use fk to find their values for the current state and the desired state to calculate cost.
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

        #with torch.no_grad():
        # target_points = torch.roll(transformed_points, shifts=1, dims=1).detach() #shifting the horizon

        # distances = torch.linalg.norm(
        #     transformed_points - target_points, dim=-1
        # )
        # #distances[:, -1, :] *= 0.0
        # distances.index_fill(dim=1, index=distances.size(1) - 1, value=0.0)
        # print("distances", distances)

        distances = torch.linalg.norm(transformed_points[:, 3:, :, :] - transformed_points[:, :-3, :, :].detach(), dim=-1) #[batch, horizon-1, 3]
        print("distances.shape", distances.shape)
        print(torch.zeros((distances.shape[0], transformed_points.shape[1]-distances.shape[1], distances.shape[2]), device=distances.device, dtype=distances.dtype).shape)
        distances = torch.cat([distances, torch.zeros((distances.shape[0], transformed_points.shape[1]-distances.shape[1], distances.shape[2]), device=distances.device, dtype=distances.dtype)], dim=1)
        print("distances.shape", distances.shape)


        cost = torch.sum(distances, dim=-1)
        print("cost.shape", cost.shape)
        #cost = cost * self.distance_scaling(cost, -10, 1000)
        cost *= 20000 #for mpc 10000
        print("cost output from the custom cost", cost.mean())

        # take minimum of symmetric cost; maybe not

        return cost * self.weight

    def forward(self, gripper_pos_batch, gripper_rot_batch):
        #return self.snake(gripper_pos_batch, gripper_rot_batch)
        #return self.cost_window(gripper_pos_batch, gripper_rot_batch)
        '''
        input:
        pos: [batch, horizon, 3]
        rot: [batch, horizon, 4]
        '''
        batch = gripper_pos_batch.shape[0]
        horizon = gripper_pos_batch.shape[1]
        # print('input shape:', gripper_pos_batch.shape, gripper_rot_batch.shape)
        #There are points that are defined as relative to the reference point on the pose. We use fk to find their values for the current state and the desired state to calculate cost.
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
        #print("cost.shape", cost.shape)
        #cost = cost * self.distance_scaling(cost, 10, 3000)
        cost *= 200 #for mpc 10000
        #cost[:, :-15] *= 0.0
        #print("cost output", cost.mean())

        # take minimum of symmetric cost; maybe not

        return cost * self.weight * 0.0
