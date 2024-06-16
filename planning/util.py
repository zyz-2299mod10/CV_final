import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *

def same_traj_dim(collision_free_traj):
    '''
    make every trajectory have same timestep (convenient to index and turn into tensor)
    '''
    max_timesteps = 0
    for timestep in collision_free_traj:
        if len(timestep) > max_timesteps:
            max_timesteps = len(timestep)

    for timestep_idx in range(len(collision_free_traj)):
        for _ in range(max_timesteps - len(collision_free_traj[timestep_idx])):
            collision_free_traj[timestep_idx].append(collision_free_traj[timestep_idx][-1])
    
    return max_timesteps, collision_free_traj

def set_curobo_cuboid(p, num_envs, error = None):
    '''
    p: isaac pose
    error: [x, y, z] offset to p
    '''    
    if error is None:
        cuboid = [[p.p.x, p.p.y, p.p.z,
                   p.r.w, p.r.x, p.r.y, p.r.z]] * num_envs

    else:
        cuboid = [[p.p.x - error[0], p.p.y - error[1], p.p.z - error[2],
                   p.r.w, p.r.x, p.r.y, p.r.z]] * num_envs
        
    return cuboid