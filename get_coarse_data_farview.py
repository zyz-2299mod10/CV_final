from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import os
import pickle
import copy
import cv2
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, sys.path[0]+"/../..")
from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import (
    orientation_error,
    SetRotationPoint, 
    quat_mul_NotForTensor,
    H_2_Transform,
    euler_xyz_to_matrix,
    pq_to_H,
    Transform_2_H
    )
from util.reconstruct import depth_image_to_point_cloud
from util.camera import compute_camera_intrinsics_matrix, world_2_camera_frame

def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2).to(device)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

def point2pixel(keypoint_in_camera, focal_x, focal_y, principal_x, principal_y):
    """
    Given keypoint in camera frame, project them into image
    space and compute the depth value expressed in [mm]
    :param keypoint_in_camera: (4, n_keypoint) keypoint expressed in camera frame in meter
    :return: (3, n_keypoint) where (xy, :) is pixel location and (z, :) is depth in mm
    """
    assert len(keypoint_in_camera.shape) == 2
    n_keypoint = keypoint_in_camera.shape[0]
    xy_depth = np.zeros((n_keypoint, 3), dtype=np.int64)
    xy_depth[:, 0] = (np.divide(keypoint_in_camera[:, 0], keypoint_in_camera[:, 2]) * focal_x + principal_x).astype(np.int64)
    xy_depth[:, 1] = (np.divide(keypoint_in_camera[:, 1], keypoint_in_camera[:, 2]) * focal_y + principal_y).astype(np.int64)
    xy_depth[:, 2] = (1000.0 * keypoint_in_camera[:, 2]).astype(np.int64)
    return xy_depth

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [ # --device,
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options only ik"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
    {"name": "--chunk_idx", "type": int, "default": 0, "help": "chunk_idx"}
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK)",
    custom_parameters=custom_parameters,
)

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'
# device = 'cpu'

# Grab controller
controller = args.controller
assert controller in {"ik"}, f"Invalid controller specified -- options only ik. Got: {controller}"

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.contact_collection = gymapi.ContactCollection(1)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "/home/hcis/isaacgym/assets"
urdf_root = "/home/hcis/YongZhe/obj-and-urdf/urdf"

# create table asset
table_dims = gymapi.Vec3(0.5, 0.5, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create usb place
usb_place = 'usb_place_my.urdf'
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
# asset_options.disable_gravity = True
usb_place_asset = gym.load_asset(sim, urdf_root, usb_place, asset_options)

usb_place_urdf_info = get_urdf_info(urdf_root, usb_place)
usb_place_aabb = usb_place_urdf_info.get_mesh_aabb_size()
usb_place_collisionMesh_path = usb_place_urdf_info.get_collision_pathName_scale()["filename"]
usb_place_scale = usb_place_urdf_info.get_collision_pathName_scale()["scale"]

# create usb
usb = "1baa93373407c8924315bea999e66ce3.urdf" 
asset_options = gymapi.AssetOptions()
# asset_options.disable_gravity = True
# asset_options.fix_base_link = True
usb_asset = gym.load_asset(sim, urdf_root, usb, asset_options)

usb_urdf_info = get_urdf_info(urdf_root, usb)
usb_aabb = usb_urdf_info.get_mesh_aabb_size()
usb_collisionMesh_path = usb_urdf_info.get_collision_pathName_scale()["filename"]
usb_scale = usb_urdf_info.get_collision_pathName_scale()["scale"]

# create usb box asset
usb_dims = gymapi.Vec3(usb_aabb[0], usb_aabb[1], usb_aabb[2])

# create usb place asset
usb_place_dims = gymapi.Vec3(usb_place_aabb[0], usb_place_aabb[1], usb_place_aabb[2]) # To fit the usb place size

# add obstacle
ob1_dim = gymapi.Vec3(0.025, 0.025, 0.04)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ob1_asset = gym.create_box(sim, ob1_dim.x, ob1_dim.y, ob1_dim.z, asset_options)

ob2_dim = gymapi.Vec3(0.025, 0.025, 0.04)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ob2_asset = gym.create_box(sim, ob2_dim.x, ob2_dim.y, ob2_dim.z, asset_options)

ob3_dim = gymapi.Vec3(0.025, 0.025, 0.04)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ob3_asset = gym.create_box(sim, ob3_dim.x, ob3_dim.y, ob3_dim.z, asset_options)

# create keypoints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
sphere_radius = 0.005
usb_top = gym.create_sphere(sim, sphere_radius, asset_options)
usb_bottom = gym.create_sphere(sim, sphere_radius, asset_options)
hole_top = gym.create_sphere(sim, sphere_radius, asset_options)
hole_bottom = gym.create_sphere(sim, sphere_radius, asset_options)
hole_obj_bottom = gym.create_sphere(sim, sphere_radius, asset_options)

# create camera
camera_props = gymapi.CameraProperties()
camera_props.width = 256
camera_props.height = 256
camera_props.horizontal_fov = 90 # default
camera_props.near_plane = 0.01
camera_props.far_plane = 1.5
camera_props.enable_tensors = True

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_mids = 0.4 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:7].fill(50.0)
franka_dof_props["damping"][:7].fill(40.0)
    
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(100.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid
chunk_idx = args.chunk_idx
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(table_pose.p.x - 0.65 * table_dims.x, table_pose.p.y, table_dims.z)

usb_pose = gymapi.Transform() # usb
box_pose = gymapi.Transform()

usb_place_pose = gymapi.Transform() # usb place
goal_pose = gymapi.Transform()

usb_top_pose = gymapi.Transform()
usb_bottom_pose = gymapi.Transform()
hole_top_pose = gymapi.Transform()
hole_bottom_pose = gymapi.Transform()
hole_obj_bottom_pose = gymapi.Transform()

ob1_pose = gymapi.Transform()
ob2_pose = gymapi.Transform()
ob3_pose = gymapi.Transform()

viewer_pos = gymapi.Vec3(table_pose.p.x + 0.5,  table_pose.p.y, 1)
viewer_target = gymapi.Vec3(table_pose.p.x, table_pose.p.y, 0.5)

# camera setting
# cam_local_transform = gymapi.Transform()
# cam_local_transform.p = gymapi.Vec3(0.1, 0, 0.05)
# # flip_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(180))
# look_down = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-90))
# cam_local_transform.r = look_down # quat_mul_NotForTensor(flip_q, look_down)
cam_pos = gymapi.Vec3(table_pose.p.x + 0.2,  table_pose.p.y, 0.7)
cam_target = gymapi.Vec3(table_pose.p.x, table_pose.p.y, 0.3)

envs = []
usb_place_idxs = []
usb_idxs = []
hand_idxs = []
ob1_idxs = []
ob2_idxs = []
ob3_idxs = []
franka_idxs = []
init_pos_list = []
init_rot_list = []

usb_top_kpts = []
usb_bottom_kpts = []
hole_top_kpts = []
hole_bottom_kpts = []
hole_obj_bottom_kpts = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

camera_handles = []

# set random seed
np.random.seed(chunk_idx)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add camera
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
    camera_handles.append(camera_handle)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    table_idx = gym.get_actor_rigid_body_index(env, table_handle, 0, gymapi.DOMAIN_SIM)

    # add usb
    usb_pose.p.x = table_pose.p.x
    usb_pose.p.y = table_pose.p.y + 0.15
    usb_pose.p.z = table_dims.z + 0.5 * usb_dims.z
    usb_handle = gym.create_actor(env, usb_asset, usb_pose, "usb", i, 1)
    usb_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    usb_color = gymapi.Vec3(1, 0, 0)
    gym.set_rigid_body_color(env, usb_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_color)
    usb_idx = gym.get_actor_rigid_body_index(env, usb_handle, 0, gymapi.DOMAIN_SIM)
    usb_idxs.append(usb_idx)

    # add usb place
    usb_place_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-2, 2) * math.pi)
    usb_place_pose.p.x = usb_pose.p.x + np.random.uniform(-0.065, 0.065)
    usb_place_pose.p.y = usb_pose.p.y + np.random.uniform(-0.2, -0.35)
    usb_place_pose.p.z = table_dims.z + usb_place_dims.z
    usb_place_handle = gym.create_actor(env, usb_place_asset, usb_place_pose, "usb place", i, 2)
    usb_place_color = gymapi.Vec3(0, 1, 0)
    gym.set_rigid_body_color(env, usb_place_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_place_color)
    usb_place_idx = gym.get_actor_rigid_body_index(env, usb_place_handle, 0, gymapi.DOMAIN_SIM)
    usb_place_idxs.append(usb_place_idx)

    # add obstacle
    ob1_pose.p.x = usb_pose.p.x + np.random.uniform(-0.05, 0.05)
    ob1_pose.p.y = usb_pose.p.y - 0.1
    ob1_pose.p.z = table_dims.z + ob1_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob1_handle = gym.create_actor(env, ob1_asset, ob1_pose, "ob1", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob1_idx = gym.get_actor_rigid_body_index(env, ob1_handle, 0, gymapi.DOMAIN_SIM)
    ob1_idxs.append(ob1_idx)
    
    ob2_pose.p.x = usb_pose.p.x + np.random.uniform(-0.05, 0.05)
    ob2_pose.p.y = usb_pose.p.y - 0.1
    ob2_pose.p.z = table_dims.z + ob2_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob2_handle = gym.create_actor(env, ob2_asset, ob2_pose, "ob2", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob2_idx = gym.get_actor_rigid_body_index(env, ob2_handle, 0, gymapi.DOMAIN_SIM)
    ob2_idxs.append(ob2_idx)
    
    ob3_pose.p.x = usb_pose.p.x + np.random.uniform(-0.05, 0.05)
    ob3_pose.p.y = usb_pose.p.y - 0.1
    ob3_pose.p.z = table_dims.z + ob3_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob3_handle = gym.create_actor(env, ob3_asset, ob3_pose, "ob3", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob3_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob3_idx = gym.get_actor_rigid_body_index(env, ob3_handle, 0, gymapi.DOMAIN_SIM)
    ob3_idxs.append(ob3_idx)

    # add usb keypoints
    top_bottom_dist = 0.015
    usb_bottom_pose.p = gymapi.Vec3(usb_pose.p.x, usb_pose.p.y - 0.035, 0.011 + usb_place_pose.p.z)
    usb_bottom_pose.r = usb_pose.r
    # usb_bottom_handle = gym.create_actor(env, usb_bottom, usb_bottom_pose, "usb_bottom", i, 1)
    # usb_bottom_color = gymapi.Vec3(0, 0, 1)
    # gym.set_rigid_body_color(env, usb_bottom_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_bottom_color)
    ubt = Transform_2_H(usb_bottom_pose)
    usb_bottom_kpts.append(ubt)

    usb_top_pose.p = gymapi.Vec3(usb_pose.p.x, usb_bottom_pose.p.y - top_bottom_dist, usb_bottom_pose.p.z)
    usb_top_pose.r = usb_pose.r
    # usb_top_handle = gym.create_actor(env, usb_top, usb_top_pose, "usb_top", i, 1)
    # gym.set_rigid_body_color(env, usb_top_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_bottom_color)
    utt = Transform_2_H(usb_top_pose)
    usb_top_kpts.append(utt)

    # add hole keypoints
    y_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -0.5 * math.pi) # make x-axis up (same as CFVS setting)
    hole_kpt_rot = quat_mul_NotForTensor(usb_place_pose.r, y_rot)

    hole_top_pose.p = gymapi.Vec3(usb_place_pose.p.x, usb_place_pose.p.y, usb_place_pose.p.z)
    hole_top_pose.r = hole_kpt_rot
    # hole_top_handle = gym.create_actor(env, hole_top, hole_top_pose, "hole_top", i, 2)
    # gym.set_rigid_body_color(env, hole_top_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_bottom_color)
    htt = Transform_2_H(hole_top_pose)
    hole_top_kpts.append(htt)

    hole_bottom_pose.p = gymapi.Vec3(usb_place_pose.p.x, usb_place_pose.p.y, usb_place_pose.p.z - top_bottom_dist)
    hole_bottom_pose.r = hole_kpt_rot
    # hole_bottom_handle = gym.create_actor(env, hole_bottom, hole_bottom_pose, "hole_bottom", i, 2)
    # gym.set_rigid_body_color(env, hole_bottom_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_bottom_color)
    hbt = Transform_2_H(hole_bottom_pose)
    hole_bottom_kpts.append(hbt)

    hole_obj_bottom_pose.p = gymapi.Vec3(usb_place_pose.p.x, usb_place_pose.p.y, usb_place_pose.p.z - usb_place_dims.z)
    hole_obj_bottom_pose.r = hole_kpt_rot
    # hole_obj_bottom_handle = gym.create_actor(env, hole_obj_bottom, hole_obj_bottom_pose, "hole_obj_bottom", i, 2)
    # gym.set_rigid_body_color(env, hole_obj_bottom_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, usb_bottom_color)
    hobt = Transform_2_H(hole_obj_bottom_pose)
    hole_obj_bottom_kpts.append(hobt)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
        
    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# point camera at middle env
viewer_pos = gymapi.Vec3(table_pose.p.x + 0.5,  table_pose.p.y, 1)
viewer_target = gymapi.Vec3(table_pose.p.x, table_pose.p.y, 0.5)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, viewer_pos, viewer_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# USB fine-grained rotation
fine_grained_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi)
fine_grained_rot = quat_mul_NotForTensor(fine_grained_rot, gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi))
fine_grained_rot = torch.stack(num_envs * [torch.tensor([fine_grained_rot.x,
                                                         fine_grained_rot.y,
                                                         fine_grained_rot.z,
                                                         fine_grained_rot.w])]).to(device).view((num_envs, 4))
# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7].to(device)

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states).to(device)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states).to(device)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1).to(device)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1).to(device)

# get dof force
gym.refresh_net_contact_force_tensor(sim)
_net_force = gym.acquire_net_contact_force_tensor(sim)
net_force = gymtorch.wrap_tensor(_net_force)

# Create a tensor noting whether the hand should return to the initial position
Findbox = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1).to(device)

print("============================= grasp ===================================")
with open("./Grasping_pose.pickle", "rb") as f:
    grasp = pickle.load(f)

z_mat = euler_xyz_to_matrix(0, 0, np.pi/2)
grasp = torch.tensor(grasp, dtype = torch.float32) @ z_mat.repeat(100, 1, 1) # 100 for sample 100 grasping

grasping_id = 10
t = H_2_Transform(grasp[grasping_id, ...])

grasp_position = [t.p.x, t.p.y, t.p.z]
grasp_pos = []
for env in range(num_envs):
    gym.refresh_rigid_body_state_tensor(sim)
    
    usb_pos_tmp = rb_states[usb_idxs[env], :3].tolist()
    grasp_pos.append([grasp_position[0] + usb_pos_tmp[0],
                      grasp_position[1] + usb_pos_tmp[1],
                      grasp_position[2] + usb_pos_tmp[2] - usb_dims.z * 0.8])   

grasp_pos = torch.tensor(grasp_pos).to(device)
grasp_rot = torch.tensor([t.r.x, t.r.y, t.r.z, t.r.w]).repeat(num_envs, 1).to(device)
print("=========================== grasp done ================================")


print("============================= lift height ===================================")
# lift_position = []
# total_dist = 0.15
# for env in range(num_envs):
#     gym.refresh_rigid_body_state_tensor(sim)
    
#     usb_pos_tmp = rb_states[usb_idxs[env], :3].tolist()
#     usb_place_pos_tmp = rb_states[usb_place_idxs[env], :3].tolist()

#     z_dist = total_dist**2 - ((usb_pos_tmp[0] - usb_place_pos_tmp[0])**2 + (usb_pos_tmp[1] - usb_place_pos_tmp[1])**2)
#     if(z_dist < 0):
#         lift_z = usb_place_pos_tmp[2]    
#     else:
#         lift_z = (z_dist)**0.5 + usb_place_pos_tmp[2]

#     lift_position.append([usb_pos_tmp[0], usb_pos_tmp[1], lift_z])
    
# lift_position = torch.tensor(lift_position).to(device)
# print(lift_position)

lift_position = torch.tensor([usb_pose.p.x, usb_pose.p.y, usb_place_pose.p.z + 0.1]).to(device)

print("=============================================================================")

def depth_2_pcd(depth, factor, K):
    xmap = np.array([[j for i in range(depth.shape[0])] for j in range(depth.shape[1])])
    # ymap = np.array([[i for i in range(depth.shape[0]-1, -1, -1)] for j in range(depth.shape[1])])
    ymap = np.array([[i for i in range(depth.shape[0])] for j in range(depth.shape[1])])
    # v, u = np.mgrid[0:depth.shape[0], depth.shape[1]-1:-1:-1]
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    mask_depth = depth < 1.5
    choose = mask_depth.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        return None

    depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / factor
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    pcd = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcd, choose


# simulation loop
# store_path = "./CFVS_data/processed"
store_path = "./HW/CV/far_view_noise/processed"

image_path = os.path.join(store_path, "images")
peg_in_hole_yaml = "peg_in_hole.yaml"

os.makedirs(f"{store_path}", exist_ok=True)
os.makedirs(f"{image_path}", exist_ok=True)
cnt = 0 # let camera run some iteration -> refresh the image
while not gym.query_viewer_has_closed(viewer):    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    
    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    if cnt == 10:
        print("save dataset!!")
        data = {}
        for env in range(num_envs):
            store_idx = chunk_idx * num_envs + env
            '''
            image
            '''
            rgb_path = str(store_idx).zfill(6) + '_rgb.png'
            depth_path = str(store_idx).zfill(6) + '_depth.png'
            color_tensor = gym.get_camera_image_gpu_tensor(sim, envs[env], camera_handles[env], gymapi.IMAGE_COLOR)
            rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[..., 0:3]
            depth_tensor = gym.get_camera_image_gpu_tensor(sim, envs[env], camera_handles[env], gymapi.IMAGE_DEPTH)
            depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
            
            # save rgb & depth image
            height, width, _ = rgb.shape
            center = (width / 2, height / 2)
            depth_mm = (-depth * 1000).astype(np.uint16) # needed to negative depth (be positive)
            
            cv2.imwrite(os.path.join(image_path, rgb_path), rgb)
            cv2.imwrite(os.path.join(image_path, depth_path), depth_mm)

            # reconstruct
            # cam_trans = torch.tensor(view_matrix, device=device)
            # rgb = torch.tensor(rgb, device=device)
            # depth = torch.tensor(depth, device=device)

            # intrinsic = compute_camera_intrinsics_matrix(height, width, 90)
            # pc, color = depth_image_to_point_cloud(rgb, depth, intrinsic, device=device)    
            # pc = cam_trans @ pc
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc.T[:, :3].cpu().numpy())
            # # pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy())                  
            # trimesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)   
            # o3d.visualization.draw_geometries([trimesh, pcd])

            # o3d.io.write_point_cloud(os.path.join(pcd_path , f"pcd{env}.pcd"), pcd)

            '''
            peg_in_hole_yaml
            '''
            # kpts in world
            peg_kpt_top_in_world = usb_top_kpts[env][:3, 3]
            peg_kpt_bottom_in_world = usb_bottom_kpts[env][:3, 3]
            hole_kpt_top_in_world = hole_top_kpts[env][:3, 3]
            hole_kpt_bottom_in_world = hole_bottom_kpts[env][:3, 3]

            hole_kpt_top_pose = hole_top_kpts[env].numpy()
            hole_kpt_obj_bottom_pose = hole_obj_bottom_kpts[env].numpy()

            # camera info
            t = gym.get_camera_transform(sim, envs[env], camera_handles[env])
            view_matrix = gym.get_camera_view_matrix(sim, envs[env], camera_handles[env])
            view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
            view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
            view_matrix[3, :3] = np.array([0, 0, 0])

            view_t = H_2_Transform(torch.tensor(view_matrix, device=device))
            cam2world = {
                'quaternion':{'x': view_t.r.x, 'y': view_t.r.y, 'z': view_t.r.z, 'w': view_t.r.w},
                'translation':{'x': view_t.p.x, 'y':view_t.p.y, 'z':view_t.p.z}
            }

            intrinsic = compute_camera_intrinsics_matrix(height, width, camera_props.horizontal_fov)
            f_x = intrinsic[0, 0].item()
            f_y = intrinsic[1, 1].item()
            p_x = intrinsic[0, 2].item()
            p_y = intrinsic[1, 2].item()

            # kpts in camera
            peg_top_in_cam = world_2_camera_frame(cam_coordinate=view_matrix, query=peg_kpt_top_in_world)
            peg_bottom_in_cam = world_2_camera_frame(cam_coordinate=view_matrix, query=peg_kpt_bottom_in_world)
            hole_top_in_cam = world_2_camera_frame(cam_coordinate=view_matrix, query=hole_kpt_top_in_world)
            hole_bottom_in_cam = world_2_camera_frame(cam_coordinate=view_matrix, query=hole_kpt_bottom_in_world)

            keypoint_in_camera = np.array(
                [peg_top_in_cam, peg_bottom_in_cam, hole_top_in_cam, hole_bottom_in_cam]
            )
            xy_depth = point2pixel(keypoint_in_camera=keypoint_in_camera,
                                    focal_x=f_x,
                                    focal_y=f_y,
                                    principal_x=p_x,
                                    principal_y=p_y
                                )
            xy_depth_list = []
            for i in range(len(xy_depth)):
                x = int(xy_depth[i][0])
                y = int(xy_depth[i][1])
                z = int(xy_depth[i][2])
                xy_depth_list.append([x, y, z])

            bbox_top_left_xy = [0, 0]
            bbox_bottom_right_xy = [height, width]

            # delta part
            gripper_pose = usb_top_kpts[env].numpy()
            cnt_xyz = gripper_pose[:3, 3]
            cnt_rot = gripper_pose[:3, :3]

            pre_xyz = hole_kpt_top_pose[:3, 3]
            pre_rot = hole_kpt_top_pose[:3, :3]

            delta_translation = pre_xyz - cnt_xyz
            delta_rotation = np.dot(cnt_rot.T, pre_rot)

            r = R.from_matrix(delta_rotation)
            r_euler = r.as_euler('zyx', degrees=True)
            delta_rotation_rel2world = np.dot(pre_rot, cnt_rot.T)
            r_rel2world = R.from_matrix(delta_rotation_rel2world)
            r_euler_rel2world = r_rel2world.as_euler('zyx', degrees=True)

            step_size = 0

            info = {'3d_keypoint_camera_frame': [[float(v) for v in peg_top_in_cam],
                                            [float(v) for v in peg_bottom_in_cam],
                                            [float(v) for v in hole_top_in_cam],
                                            [float(v) for v in hole_bottom_in_cam ]],
            'hole_keypoint_obj_top_pose_in_world': hole_kpt_top_pose.tolist(),
            'hole_keypoint_obj_bottom_pose_in_world': hole_kpt_obj_bottom_pose.tolist(),
            'bbox_bottom_right_xy': bbox_bottom_right_xy,
            'bbox_top_left_xy': bbox_top_left_xy,
            'camera_to_world': cam2world,
            'camera_matrix': [view_matrix.tolist()],
            'depth_image_filename': [depth_path],  # list
            'rgb_image_filename': [rgb_path],  # list
            'keypoint_pixel_xy_depth': xy_depth_list,
            'delta_rotation_matrix': delta_rotation.tolist(),
            'delta_translation': delta_translation.tolist(),
            'gripper_pose': gripper_pose.tolist(),
            'step_size': step_size,
            'r_euler': r_euler.tolist(),
            'r_euler_rel2world': r_euler_rel2world.tolist(),
            }

            data[store_idx] = info   

        with open(os.path.join(store_path, peg_in_hole_yaml), "a+") as f:
            yaml.dump(data, f)        
        break
    cnt += 1

    # update viewer
    gym.end_access_image_tensors(sim)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
