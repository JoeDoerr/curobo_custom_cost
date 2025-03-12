import mujoco
import mujoco.viewer

import time
import numpy as np

from tracikpy import TracIKSolver

import rospy
import rospkg

import transformations as tf

import torch

def sim_configs_mujoco(model, data, poses, qpos_inds, ref_frames=[], tlim=5):
    i = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            if time.time() - start >= tlim:
                i += 1
                if i >= len(poses):
                    viewer.close()
                    time.sleep(1)

                start = time.time()
                
            step_start = time.time()
            data.qpos[qpos_inds] = poses[i]
            mujoco.mj_step1(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - \
                (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def axis_angle_matrix(axis=np.array([0, 0, 1]), angle=None):
    if angle is None:
        angle = 2 * np.pi * np.random.rand() 

    quat = np.concatenate([[np.cos(angle)], axis * np.sin(angle)])

    return tf.quaternion_matrix(quat)

def get_objq_indices(obj_name):
    jnt = model.joint(model.body(obj_name).jntadr[0])
    qpos_inds = np.array(
        range(jnt.qposadr[0], jnt.qposadr[0] + len(jnt.qpos0))
    )
    return qpos_inds

def get_qpos_indices(joints):
    qpos_inds = np.array([model.joint(j).qposadr[0] for j in joints])
    return qpos_inds

def get_qvel_indices(joints):
    qvel_inds = np.array([model.joint(j).dofadr[0] for j in joints])
    return qvel_inds

def get_ctrl_indices(joints, prefix='', replace=''):
    ctrl_name = lambda j: prefix + j.replace('_joint', replace)
    ctrl_inds = [model.actuator(ctrl_name(j)).id for j in joints]
    return np.array(ctrl_inds)

def get_act_indices(joints, prefix='', replace=''):
    act_name = lambda j: prefix + j.replace('_joint', replace)
    act_inds = [model.actuator(act_name(j)).actadr[0] for j in joints]
    return np.array(act_inds)

def get_jnt_indices(joints):
    jnt_inds = np.array([model.joint(j).id for j in joints])
    return jnt_inds

rp = rospkg.RosPack()
try:
    pracsys_vbnpm_path = rp.get_path('pracsys_vbnpm')
    motoman_sda10f_path = rp.get_path('motoman_sda10f_moveit_config')
except rospkg.common.ResourceNotFound:
    pracsys_vbnpm_path = "/data/local/kc1317/workspace/src/pracsys_vbnpm/"
    motoman_sda10f_path = "/data/local/kc1317/workspace/src/motoman/motoman_sda10f_moveit_config/"
urdf = motoman_sda10f_path + '/config/gazebo_motoman_sda10f.urdf'
ik_CAMERA = TracIKSolver(urdf, "base_link", "motoman_right_ee")

mjcf = pracsys_vbnpm_path + '/tests/ycb_02_non_perishables.xml'

model = mujoco.MjModel.from_xml_path(mjcf)
data = mujoco.MjData(model)

ind = 1
poses = np.load('/data/local/kc1317/workspace/views.npy')

num_poses = poses.shape[0]

# pose = np.array([[-0.07330063, -0.11816157,  0.99028524 , 0.60334253],
#                 [-0.11816157,  0.98699138 , 0.10902226, -0.47715496],
#                 [-0.99028524, -0.10902226 ,-0.08630925,  1.02149768],
#                 [ 0.,          0.,          0.,          1.        ],])
# pose = np.array([[ 0.0972508,  -0.09938526, -0.99028524,  0.60334252],
#                 [-0.09938526,  0.9890585,  -0.10902226, -0.47715496],
#                 [ 0.99028524,  0.10902226,  0.0863093,   1.02149768],
#                 [ 0.,          0.,          0.,          1.        ]])
# qpos = np.concatenate([pose[:3, 3], tf.quaternion_from_matrix(pose)])

# print(data.body("body_cam_arm"))
tolerances = {
    'bx': 0,
    'by': 0,
    'bz': 0
}

randomize = False
configs = []
for i in range(1000):
    # pose = tf.quaternion_matrix(quat[i % num_poses].numpy())
    # pose[:3, 3] = pos[i % num_poses]
    pose = poses[i % num_poses]
    if not randomize:
        temp_pose = pose
    else:
        temp_pose = pose @ axis_angle_matrix()
    q = ik_CAMERA.ik(temp_pose, **tolerances)
    if q is not None:
        configs.append(q)

qpos_inds = get_qpos_indices(ik_CAMERA.joint_names)

print(q)
print(qpos_inds)

sim_configs_mujoco(model, data, configs, qpos_inds, tlim=3)