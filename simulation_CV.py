from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import os
import yaml
import pickle
import copy
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from planning.util import same_traj_dim, set_curobo_cuboid
from util.read_urdf import get_urdf_info
from util.other_isaacgym_fuction import (
    orientation_error,
    SetRotationPoint, 
    quat_mul_NotForTensor,
    H_2_Transform,
    euler_xyz_to_matrix,
    pq_to_H
    )
from util.camera import compute_camera_intrinsics_matrix
from hole_estimation.predict_hole_pose import CoarseMover

def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2).to(device)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

def get_hole_pose(model_path, check_point_num,
                  depth, intrinsic, view_matrix, mode, visualize = False
                  ):
    '''
    arg:
        model path: 'kpts/2022-02-??_??-??'
    
    return:
        tensor
    '''

    predictor = CoarseMover(model_path=model_path, model_name='pointnet2_kpts',
                               checkpoint_name=f'best_model_e_{check_point_num}.pth', use_cpu=False, out_channel=9)
    H = predictor.predict_kpts_pose(depth=depth, factor=1, K=intrinsic.cpu().numpy(), view_matrix=view_matrix,
                                    mode=mode, visualize=visualize)
    H = torch.tensor(H, device = device).to(torch.float32)    
    usb_place_pq = H_2_Transform(H)

    HandShouldRot = usb_place_pq.r
    HandShouldRot.w = -HandShouldRot.w
    end_rot = quat_mul(quat_conjugate(torch.tensor([[HandShouldRot.x,
                                                    HandShouldRot.y,
                                                    HandShouldRot.z,
                                                    HandShouldRot.w]], device=device)), fine_grained_rot)
    end_pos = H[:3, -1].view(1,3)

    return end_pos, end_rot

def add_gaussian_noise(img, sigma = 0.001, mean = 0):
    '''
    add gaussian noise to depth img
    args:
        img: ndarray (mm)
    '''
    max_depth = np.max(img)
    min_depth = np.min(img)
    # normalize
    img = (img - min_depth) / (max_depth - min_depth)

    noise = np.random.normal(mean, sigma, img.shape)
    out = img + noise
    out = np.clip(out, 0, 1)
    
    out = np.uint16(out*(max_depth - min_depth)) + min_depth
    noise = np.uint16(noise*(max_depth - min_depth)) + min_depth

    return out , noise

# set random seed
np.random.seed(500)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [ # --device,
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options only ik"},
    {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
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
# asset_options.fix_base_link = True
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
franka_dof_props["stiffness"][:7].fill(100.0)
franka_dof_props["damping"][:7].fill(50.0)
    
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

ob1_pose = gymapi.Transform()
ob2_pose = gymapi.Transform()
ob3_pose = gymapi.Transform()

viewer_pos = gymapi.Vec3(table_pose.p.x + 0.5,  table_pose.p.y, 1)
viewer_target = gymapi.Vec3(table_pose.p.x, table_pose.p.y, 0.5)

# camera setting
cam_local_transform = gymapi.Transform()
cam_local_transform.p = gymapi.Vec3(0.1, 0, 0.05)
# flip_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(180))
look_down = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-90))
cam_local_transform.r = look_down # quat_mul_NotForTensor(flip_q, look_down)

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

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

camera_close_view_handles = []
camera_far_view_handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)    

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
    usb_place_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-1.5, 0) * math.pi)
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
    ob1_pose.p.y = usb_pose.p.y - 0.065
    ob1_pose.p.z = table_dims.z + ob1_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob1_handle = gym.create_actor(env, ob1_asset, ob1_pose, "ob1", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob1_idx = gym.get_actor_rigid_body_index(env, ob1_handle, 0, gymapi.DOMAIN_SIM)
    ob1_idxs.append(ob1_idx)
    
    ob2_pose.p.x = usb_pose.p.x + np.random.uniform(-0.05, 0.05)
    ob2_pose.p.y = usb_pose.p.y - 0.065
    ob2_pose.p.z = table_dims.z + ob2_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob2_handle = gym.create_actor(env, ob2_asset, ob2_pose, "ob2", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob2_idx = gym.get_actor_rigid_body_index(env, ob2_handle, 0, gymapi.DOMAIN_SIM)
    ob2_idxs.append(ob2_idx)
    
    ob3_pose.p.x = usb_pose.p.x + np.random.uniform(-0.05, 0.05)
    ob3_pose.p.y = usb_pose.p.y - 0.065
    ob3_pose.p.z = table_dims.z + ob3_dim.z * 0.5 + np.random.uniform(0, 0.1)
    ob3_handle = gym.create_actor(env, ob3_asset, ob3_pose, "ob3", i, 0)
    ob_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, ob3_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, ob_color)
    ob3_idx = gym.get_actor_rigid_body_index(env, ob3_handle, 0, gymapi.DOMAIN_SIM)
    ob3_idxs.append(ob3_idx)

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

    # attach camera to hand
    camera_close_view_handle = gym.create_camera_sensor(env, camera_props)
    gym.attach_camera_to_body(camera_close_view_handle, env, hand_handle, cam_local_transform, gymapi.FOLLOW_TRANSFORM)
    camera_close_view_handles.append(camera_close_view_handle)

    # add far view camera
    camera_far_view_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_far_view_handle, env, cam_pos, cam_target)
    camera_far_view_handles.append(camera_far_view_handle)

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
new_pos_action = torch.zeros_like(dof_pos).squeeze(-1).to(device)

# -------------------------------| compute end point |---------------------------------------
# common setting
goal_offset = 0.1 # upper to the goal_pose.z

ori_usb_pos = []
end_usb_pos = []
for env in range(num_envs):
    gym.refresh_rigid_body_state_tensor(sim)
    
    goal_pos_tmp = rb_states[usb_place_idxs[env], :3].tolist()
    goal_rot_tmp = rb_states[usb_place_idxs[env], 3:7].tolist()
    
    usb_pos_tmp = rb_states[usb_idxs[env], :3].tolist()
    usb_rot_tmp = rb_states[usb_idxs[env], 3:7].tolist()
    
    start = [usb_pos_tmp[0], usb_pos_tmp[1], usb_pos_tmp[2]]
    endpos = [goal_pos_tmp[0], goal_pos_tmp[1], goal_pos_tmp[2] + goal_offset]
    inipos = [goal_pos_tmp[0], goal_pos_tmp[1], goal_pos_tmp[2]]
    end = SetRotationPoint(inipos, endpos, goal_rot_tmp)
    
    ori_usb_pos.append(start)
    end_usb_pos.append(end)
    
    print(f"env id: {env}")
    print(f"{str('start:').ljust(8)} {start}")
    print(f"{str('goal:').ljust(8)} {inipos}")
    print(f"{str('end:').ljust(8)} {end}") # end point: contact point of usb (not for franka hand) 
    print()
    
end_usb_pos = torch.tensor(end_usb_pos).view(num_envs, 3).to(device)
ori_usb_pos = torch.tensor(ori_usb_pos).view(num_envs, 3).to(device)
#------------------------------------------------------------------------------------------

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
print("=========================== grasp done =================================")

# simulation loop
curobo_yaml_gener_lock = False
far_view_lock = False
close_view_lock = False
current_idx = 0
visualize = False
while not gym.query_viewer_has_closed(viewer):    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    usb_pos = rb_states[usb_idxs, :3]
    usb_rot = rb_states[usb_idxs, 3:7]
    
    # the direction of goal rot is reverse to the hand should rot 
    # ex. goal rot = 0.4 pi , hand rot should be -0.4 pi
    goal_pos = rb_states[usb_place_idxs, :3]
    goal_rot = rb_states[usb_place_idxs, 3:7]
    
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < usb_dims.x + 0.001)
    
    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    Findbox = (Findbox & (init_dist > 0.02)).squeeze(-1)
    IsGrasp = (Findbox | gripped.squeeze(-1)).unsqueeze(-1)

    if not far_view_lock:
        # get depth image
        color_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_far_view_handles[0], gymapi.IMAGE_COLOR)
        rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[..., 0:3]
        depth_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_far_view_handles[0], gymapi.IMAGE_DEPTH)
        depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        depth = (-depth * 1000).astype(np.uint16)

        # hole pose estimate
        height, width, _ = rgb.shape
        intrinsic = compute_camera_intrinsics_matrix(height, width, camera_props.horizontal_fov)
        t = gym.get_camera_transform(sim, envs[0], camera_far_view_handles[0])
        view_matrix = gym.get_camera_view_matrix(sim, envs[0], camera_far_view_handles[0])
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
        view_matrix[3, :3] = np.array([0, 0, 0])

        depth, noise = add_gaussian_noise(depth, 0.002, 0)
        depth = depth / 1000
        end_pos_far_view, end_rot_far_view = get_hole_pose(model_path = './kpts/2024-06-07_01-59', check_point_num = 30,
                                         depth = depth, intrinsic = intrinsic, view_matrix = view_matrix, mode = 'far', visualize = visualize)
        end_pos_far_view[:, 1] = end_pos_far_view[:, 1] + 0.12
        end_pos_far_view[:, 2] = end_pos_far_view[:, 2] + 0.03
        print('coarse view', end_pos_far_view)

        far_view_lock = True
    
    if not IsGrasp.all() and not curobo_yaml_gener_lock:
        '''
        Grasp
        '''
        end_pos = copy.deepcopy(grasp_pos)
        end_rot = copy.deepcopy(grasp_rot)

        # compute position and orientation error
        pos_err = torch.where(IsGrasp, end_pos - usb_pos, end_pos - hand_pos)
        orn_err = orientation_error(end_rot, hand_rot)

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    
        # Deploy control
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)

        # gripper actions depend on distance and orientation error between hand and given grasp pose
        to_grasp_position = grasp_pos - hand_pos
        GraspPose2HandPoseDist = torch.norm(to_grasp_position, dim=-1).unsqueeze(-1).to(device)
        hand_grasp_rot_error = orientation_error(grasp_rot, hand_rot)
        HandGraspRotError= torch.norm(hand_grasp_rot_error, dim=-1).unsqueeze(-1).to(device)
        
        close_gripper = ((GraspPose2HandPoseDist < 0.005) & (HandGraspRotError < 0.005)) | gripped
        GripperOpenScale = 0.04
        keep_going = torch.logical_not(Findbox)
        close_gripper = close_gripper & keep_going.unsqueeze(-1)
        grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[GripperOpenScale, GripperOpenScale]] * num_envs).to(device))
        pos_action[:, 7:9] = grip_acts
        
        # Deploy actions
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))    
    
    elif IsGrasp.all() and not curobo_yaml_gener_lock:
        '''
        Generate collision-free trajectory (far-view)
        '''

        # for quaternion: curobo is W-X-Y-Z (IsaacGym is X-Y-Z-W)
        
        # debug
        # end_pos = end_usb_pos
        # HandShouldRot_t = copy.deepcopy(goal_rot)
        # HandShouldRot_t[:, -1] = -HandShouldRot_t[:, -1]
        # end_rot = quat_mul(quat_conjugate(HandShouldRot_t), fine_grained_rot)
        #

        end_pos = end_pos_far_view.tolist()
        hand_goal_position = end_pos
        hand_goal_position[0][-1] = hand_goal_position[0][-1] + 0.11 # hand offset to peg

        hand_rot_tmp = hand_rot[0, ...].tolist()
        hand_goal_quaternion = [[hand_rot_tmp[-1], hand_rot_tmp[0], hand_rot_tmp[1], hand_rot_tmp[2]]]
                                
        #start state
        _dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(device)
        start_position = dof_states[:, 0].view(num_envs, 1, 9).to(device)
        start_position = start_position[:, :, :7].tolist() # joint 1~7 (exclude finger)
                
        #usb place pose
        usb_place_pose_curobo = torch.cat((goal_pos, goal_rot[:, -1].view(num_envs, 1), goal_rot[:, 0:3]), 1).tolist()
        usb_pose_curobo = torch.cat((usb_pos, usb_rot[:, -1].view(num_envs, 1), usb_rot[:, 0:3]), 1).tolist() 
        
        #table pose
        table_pose_curubo = set_curobo_cuboid(table_pose, num_envs, error = [0, 0, 0.05]) # make table down a little bit (avoid detect collision in start state)
        ob1_pose_curobo = set_curobo_cuboid(ob1_pose, num_envs)
        ob2_pose_curobo = set_curobo_cuboid(ob2_pose, num_envs)
        ob3_pose_curobo = set_curobo_cuboid(ob3_pose, num_envs)
                                
        curobo_config_path = "./planning/yaml/curobo_config.yaml"
        curobo_trajectory_path = "./planning/yaml/curobo_trajectory.yaml"
        
        curobo_config = {
            "trajectory_path": curobo_trajectory_path,
            "num_envs": num_envs,
            "robot_type": "franka",
            "robot_position": [franka_pose.p.x, franka_pose.p.y, franka_pose.p.z], # 1 dimension
                                
            "cuboid":{ # (num_env, -1)
                "table":{
                    "dims": [table_dims.x, table_dims.y, table_dims.z], # 1 dimension
                    "pose": table_pose_curubo,
                    },
                
                "ob1":{
                    "dims": [ob1_dim.x, ob1_dim.y, ob1_dim.z],
                    "pose": ob1_pose_curobo,
                },
                "ob3":{
                    "dims": [ob2_dim.x, ob2_dim.y, ob2_dim.z],
                    "pose": ob2_pose_curobo,
                },
                "ob3":{
                    "dims": [ob3_dim.x, ob3_dim.y, ob3_dim.z],
                    "pose": ob3_pose_curobo,
                }
            },

            "mesh":{ # (num_env, -1) 
                "usb_place":{
                    "file_path": usb_place_collisionMesh_path,
                    "pose": usb_place_pose_curobo,
                    "scale": usb_place_scale,
                },
                
                "usb":{
                    "file_path": usb_collisionMesh_path,
                    "pose": usb_pose_curobo,
                    "scale": usb_scale,
                }
            },
                
            # (num_env, -1) 
            "start_state": start_position,
            "goal_position": hand_goal_position,
            "goal_quaternion":  hand_goal_quaternion
        }            
            
        with open(curobo_config_path, 'w') as f:
            yaml.dump(curobo_config, f)
                        
        # generate trajectory
        os.system("bash ./sim_tmp.sh")
               
        with open(curobo_trajectory_path, "r") as f:
            collision_free_traj = yaml.safe_load(f)["traj"]
        
        max_timesteps, collision_free_traj = same_traj_dim(collision_free_traj)
                
        collision_free_traj = torch.tensor(collision_free_traj).to(device)
        
        curobo_yaml_gener_lock = True
    
    elif(curobo_yaml_gener_lock):
        '''
        execute trajectory
        '''
        grip_acts = torch.tensor([[[0., 0.]]] * num_envs).view(num_envs, 2).to(device) # (num_envs, timesteps, joints)        
        pos_action = torch.cat((collision_free_traj[:, current_idx, :], grip_acts), 1).to(device) 
            
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))        
        
        _dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states = gymtorch.wrap_tensor(_dof_states).to(device)
        robot_js = dof_states[:, 0].view(num_envs, 1, 9).to(device)              
        joint_diff = torch.norm((collision_free_traj[:, current_idx, :] - robot_js[:, :, :7]), dim=-1).unsqueeze(-1).to(device)
        if current_idx == max_timesteps - 1: current_idx = current_idx
        elif (joint_diff < 0.1).all(): current_idx += 1
    
    to_far_vew_dist = usb_pos - end_pos_far_view
    ToFarViewDist = torch.norm(to_far_vew_dist, dim=-1).unsqueeze(-1).to(device)
    AchieveFarView = (ToFarViewDist < 0.03)

    if AchieveFarView.all() and not close_view_lock:
        # get depth image
        color_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_close_view_handles[0], gymapi.IMAGE_COLOR)
        rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[..., 0:3]
        depth_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_close_view_handles[0], gymapi.IMAGE_DEPTH)
        depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        depth = (-depth * 1000).astype(np.uint16)

        # hole pose estimate
        height, width, _ = rgb.shape
        intrinsic = compute_camera_intrinsics_matrix(height, width, camera_props.horizontal_fov)
        t = gym.get_camera_transform(sim, envs[0], camera_close_view_handles[0])
        view_matrix = gym.get_camera_view_matrix(sim, envs[0], camera_close_view_handles[0])
        view_matrix[:3, :3] = view_matrix[:3, :3] @ R.from_euler("XYZ", np.array([np.pi, 0, 0])).as_matrix()
        view_matrix[:3, 3] = np.array([t.p.x, t.p.y, t.p.z]) 
        view_matrix[3, :3] = np.array([0, 0, 0])

        depth, noise = add_gaussian_noise(depth, 0.002, 0)
        depth = depth / 1000
        end_pos_close_view, end_rot_close_view = get_hole_pose(model_path = './kpts/2024-05-29_03-56', check_point_num = 100,
                                         depth = depth, intrinsic = intrinsic, view_matrix = view_matrix, mode = 'close', visualize = visualize)

        end_pos_close_view[:, -1] = end_pos_close_view[:, -1] + 0.1
        print('close view position: ', end_pos_close_view)
        print('close view rotation: ', end_rot_close_view)
        
        close_view_lock = True
    
    elif close_view_lock:

        end_pos = copy.deepcopy(end_pos_close_view)
        end_rot = copy.deepcopy(end_rot_close_view)

        # debug
        # end_pos = end_usb_pos
        # HandShouldRot_t = copy.deepcopy(goal_rot)
        # HandShouldRot_t[:, -1] = -HandShouldRot_t[:, -1]
        # end_rot = quat_mul(quat_conjugate(HandShouldRot_t), fine_grained_rot)
        #        

        # compute position and orientation error
        pos_err = end_pos - usb_pos
        orn_err = orientation_error(end_rot, usb_rot)

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    
        # Deploy control
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
        grip_acts = torch.Tensor([[0., 0.]] * num_envs).to(device)
        pos_action[:, 7:9] = grip_acts
        
        # Deploy actions
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action)) 
        
    # update viewer
    gym.end_access_image_tensors(sim)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)    

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
