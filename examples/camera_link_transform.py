import mujoco
import mujoco.viewer

import time
import numpy as np

from tracikpy import TracIKSolver

import rospy
import rospkg

import transformations as tf

import torch

rp = rospkg.RosPack()
try:
    pracsys_vbnpm_path = rp.get_path('pracsys_vbnpm')
    motoman_sda10f_path = rp.get_path('motoman_sda10f_moveit_config')
except rospkg.common.ResourceNotFound:
    pracsys_vbnpm_path = "/data/local/kc1317/workspace/src/pracsys_vbnpm/"
    motoman_sda10f_path = "/data/local/kc1317/workspace/src/motoman/motoman_sda10f_moveit_config/"
urdf = motoman_sda10f_path + '/config/gazebo_motoman_sda10f.urdf'
ik = TracIKSolver(urdf, "base_link", "arm_right_link_7_t")

mjcf = pracsys_vbnpm_path + '/tests/ycb_02_non_perishables.xml'

model = mujoco.MjModel.from_xml_path(mjcf)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

default_link_pose = ik.fk(np.zeros(ik.number_of_joints)).astype(np.double)
default_camera = data.body("body_cam_arm")
default_camera_pose = np.eye(4)
default_camera_pose[:3, :3] = np.reshape(default_camera.xmat, (3, 3))
default_camera_pose[:3, 3] = default_camera.xpos

print("EE:", default_link_pose)
print("CAMERA:", default_camera_pose)

# T_cam^w @ T_w^link = T_cam^link
relative_transform = np.linalg.pinv(default_camera_pose) @ default_link_pose
# relative_transform = np.linalg.pinv(default_link_pose) @ default_camera_pose
# relative_transform = np.linalg.pinv(relative_transform)

# zeros_mask = abs(relative_transform) < 1e-9

# relative_transform *= zeros_mask

print("REL TRANSFORM:", relative_transform)
