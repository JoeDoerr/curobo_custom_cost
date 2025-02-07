#####
# Big thanks to Peter Mitrano for this collision debugging code!
# https://github.com/NVlabs/curobo/discussions/223#discussioncomment-9143124
#####

import time

import torch
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import WorldCollisionConfig, CollisionCheckerType

# CuRobo
from curobo.rollout.arm_base import ArmBase, ArmBaseConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    get_task_configs_path,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    setup_curobo_logger("warn")
    robot_file = "motoman.yml"
    world_file = "collision_test.yml"

    tensor_args = TensorDeviceType()

    while True:
        try:
            robot_data = load_yaml(
                join_path(get_robot_configs_path(), robot_file)
            )["robot_cfg"]
        except (ParserError, ScannerError):
            print("Error parsing robot file")
            continue
        robot_data["kinematics"]["collision_sphere_buffer"] = 0.0
        robot_cfg = RobotConfig.from_dict(robot_data)
        ik_config_data = load_yaml(
            join_path(get_task_configs_path(), "gradient_ik.yml")
        )
        base_config_data = load_yaml(
            join_path(get_task_configs_path(), "base_cfg.yml")
        )
        base_config_data["world_collision_checker_cfg"][
            "checker_type"] = CollisionCheckerType.MESH
        world_model = load_yaml(join_path(get_world_configs_path(), world_file))

        world_coll_cfg = WorldCollisionConfig.load_from_dict(
            base_config_data["world_collision_checker_cfg"], world_model,
            tensor_args
        )
        world_coll_checker = create_collision_checker(world_coll_cfg)

        cfg = ArmBaseConfig.from_dict(
            robot_cfg=robot_cfg,
            model_data_dict=ik_config_data["model"],
            cost_data_dict=base_config_data["cost"],
            constraint_data_dict=base_config_data["constraint"],
            convergence_data_dict=base_config_data["convergence"],
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        rollout = ArmBase(cfg)

        kin_cfg = robot_cfg.kinematics.kinematics_config
        samples = torch.zeros(
            [100, kin_cfg.n_dof], **tensor_args.as_torch_dict()
        )
        state = rollout.dynamics_model.forward(
            rollout.start_state, samples.unsqueeze(1)
        )
        state.robot_spheres.requires_grad = True
        rollout.robot_self_collision_constraint.forward(state.robot_spheres)
        colliding_sphere_indicators = (
            rollout.robot_self_collision_constraint._sparse_sphere_idx
            .squeeze(1)
        )

        offenses = colliding_sphere_indicators.sum(dim=0)
        offending_indices = torch.argwhere(offenses).squeeze()

        if offenses.sum() == 0:
            continue

        offending_link_indices = kin_cfg.link_sphere_idx_map[offending_indices]

        link_idx_to_name_map = {
            v: k
            for k, v in kin_cfg.link_name_to_idx_map.items()
        }
        offending_link_names = [
            link_idx_to_name_map[idx.item()] for idx in offending_link_indices
        ]

        print(offenses)
        print(offending_link_names)


if __name__ == "__main__":
    main()
