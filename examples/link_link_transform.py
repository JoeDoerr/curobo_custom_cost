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

link_0 = "arm_right_link_7_t"
link_1 = "motoman_right_ee"

ik_0 = TracIKSolver(urdf, "base_link", link_0)
ik_1 = TracIKSolver(urdf, "base_link", link_1)

link_0_pose = ik_0.fk(np.zeros(ik_0.number_of_joints)).astype(np.double)
link_1_pose = ik_1.fk(np.zeros(ik_1.number_of_joints)).astype(np.double)

print("LINK 0:", link_0_pose)
print("LINK 1:", link_1_pose)

# T_cam^w @ T_w^link = T_cam^link
relative_transform = np.linalg.pinv(link_0_pose) @ link_1_pose
# relative_transform = np.linalg.pinv(relative_transform)

# zeros_mask = abs(relative_transform) < 1e-9

# relative_transform *= zeros_mask

print("REL TRANSFORM:", relative_transform)