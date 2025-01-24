#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
""" Example computing collisions using curobo

"""
# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer

if __name__ == "__main__":
    robot_file = "franka.yml"
    world_file = "collision_test.yml"
    tensor_args = TensorDeviceType()
    # config = RobotWorldConfig.load_from_config(robot_file, world_file, pose_weight=[10, 200, 1, 10],
    #                                           collision_activation_distance=0.0)
    # curobo_fn = RobotWorld(config)
    robot_file = "franka.yml"
    world_config = {
        "cuboid": {
            "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, 0.3, 1, 0, 0, 0]},
            "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
        },
        "mesh": {
            "scene": {
                "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
                "file_path": "scene/nvblox/srl_ur10_bins.obj",
            }
        },
    }
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.0
    )
    curobo_fn : RobotWorld = RobotWorld(config)

    q_sph = torch.randn((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    q_sph[..., 3] = 0.2
    d = curobo_fn.get_collision_distance(q_sph)
    print(d)

    q_s = curobo_fn.sample(5, mask_valid=False)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
    print("Collision Distance:")
    print("World:", d_world)
    print("Self:", d_self)

    spheres = 10
    length = 1
    radius = 0.1
    # CAMERA POSE
    x_camera = torch.eye(4)
    
    # Compute vector of sphere center offsets
    offsets = torch.zeros((spheres, 3))
    offsets[:, 2] = torch.linspace(0, length, spheres)

    # Compute sphere centers 
    sphere_centers = x_camera[:3, 3][torch.newaxis, :] + offsets @ x_camera[:3, :3].T
    
    radius_col = torch.full((sphere_centers.size(0), 1), radius)

    spheres = torch.cat((sphere_centers, radius_col), dim=1)[torch.newaxis, torch.newaxis, ...]

    spheres = spheres.to(device='cuda:0').contiguous()

    print(spheres.shape)

    buffer = CollisionQueryBuffer.initialize_from_shape(spheres.shape, tensor_args, collision_types = curobo_fn.world_model.collision_types)

    weight = torch.tensor([1.0], device='cuda:0')
    activation_distance = torch.tensor([0.0], device='cuda:0')

    print("spheres stride", spheres.stride())

    dist = curobo_fn.world_model.get_sphere_collision(spheres.contiguous(), buffer, weight, activation_distance)
    
    print(dist)
    print(dist.shape)
