import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


tensor_args = TensorDeviceType()
world_file = "collision_shelf.yml"
robot_file = "motoman.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_file,
    tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    use_cuda_graph=True,
    num_trajopt_seeds=12,
    num_graph_seeds=1,
    num_ik_seeds=30,
    trajopt_dt=0.05,
    trajopt_tsteps=150,
    optimize_dt=False,
    # trim_steps=[1, None],
    interpolation_dt=0.05,
    # interpolation_type=InterpolateType.CUBIC,
    # collision_cache=collision_cache,
    # collision_activation_distance=0.02,
    # collision_max_outside_distance=0.02,
    # collision_checker_type=CollisionCheckerType.MESH,
    trajopt_fix_terminal_action=True,
    # partial_ik_opt=True,
)

motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()
state = motion_gen.rollout_fn.compute_kinematics(
    JointState.from_position(retract_cfg.view(1, -1))
)

motion_gen.warmup(n_goalset=36)

start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.6)

state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

goal_pose = Pose(
    state.ee_pos_seq.repeat(2, 1).view(1, -1, 3),
    quaternion=state.ee_quat_seq.repeat(2, 1).view(1, -1, 4),
)
goal_pose.position[0, 0, 0] -= 0.1

start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

m_config = MotionGenPlanConfig(timeout=2, enable_finetune_trajopt=False)#False, True)#, num_trajopt_seeds=10)

initial_test = False
if initial_test:
    result = motion_gen.plan_goalset(start_state, goal_pose, m_config)

    print("Initial goalset query result")
    print(result.success, result.status)

ind = 0
pos = torch.load(f'/data/local/kc1317/workspace/pos{ind}.pt').to(device=tensor_args.device).contiguous()
quat = torch.load(f'/data/local/kc1317/workspace/quat{ind}.pt').to(device=tensor_args.device).contiguous()

print(pos.shape, quat.shape)

ray_goal = Pose(
    pos,
    quat,
)
result_ray_goals = motion_gen.plan_goalset(start_state, ray_goal, m_config)

print("Ray goalset query result")
print(result_ray_goals.success, result_ray_goals.status)

for i, (p, q) in enumerate(zip(pos, quat)):
    print(p, q)
    temp_ray_goal = Pose(
        p, q
    )

    single_result = motion_gen.plan_single(start_state, temp_ray_goal, m_config)

    print(f"Ray {i} query result")
    print(result_ray_goals.success, result_ray_goals.status)