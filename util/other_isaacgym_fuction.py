from isaacgym import gymapi
from isaacgym.torch_utils import *

from se3dif.utils import SO3_R3
import theseus as th

import math
import torch
from pytorch3d.transforms import (
    matrix_to_quaternion, 
    euler_angles_to_matrix,
)

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) + 0.2 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

def slerp(q1, q2, t):
    """Spherical linear interpolation for batches of quaternions."""
    q1_norm = torch.norm(q1, dim=1, keepdim=True)
    q2_norm = torch.norm(q2, dim=1, keepdim=True)

    q1 = q1 / q1_norm  # Normalize quaternion
    q2 = q2 / q2_norm

    dot = torch.sum(q1 * q2, dim=1, keepdim=True)

    q2 *= torch.sign(dot)  # Ensure shortest path
    dot *= torch.sign(dot)

    DOT_THRESHOLD = 0.9995
    lerp_flag = dot > DOT_THRESHOLD

    inv_sin_theta_0 = 1 / torch.sqrt(1 - dot * dot)
    theta_0 = torch.acos(dot)
    theta = theta_0 * t

    s0 = torch.cos(theta) - dot * torch.sin(theta) * inv_sin_theta_0
    s1 = torch.sin(theta) * inv_sin_theta_0
    q_slerp = (s0 * q1 + s1 * q2)

    # Perform lerp for close quaternions
    q_lerp = (1 - t) * q1 + t * q2
    q_lerp = q_lerp / torch.norm(q_lerp, dim=1, keepdim=True)

    q_slerp = torch.where(lerp_flag, q_lerp, q_slerp)

    return q_slerp

def SetRotationPoint(inipos, endpos, rotation):
    endOri = gymapi.Vec3(endpos[0], endpos[1], endpos[2])
    rot = gymapi.Quat(rotation[0], rotation[1], rotation[2], rotation[3])
    endR = gymapi.Quat.rotate(rot, gymapi.Vec3(endOri.x - inipos[0], endOri.y - inipos[1], endOri.z - inipos[2]))
    end = [endR.x + inipos[0], endR.y + inipos[1] , endR.z + inipos[2]]

    return end

def Set_box_oob(ob, ob_dim, rotation):    
    obxmin = ob[0] - ob_dim.x * 0.5
    obxmax = ob[0] + ob_dim.x * 0.5
    obymin = ob[1] - ob_dim.y * 0.5
    obymax = ob[1] + ob_dim.y * 0.5
    obzmin = ob[2] - ob_dim.z * 0.5
    obzmax = ob[2] + ob_dim.z * 0.5
    
    ob_scale = [obxmin, obxmax, obymin, obymax, obzmin, obzmax]
    ob_vertex = [[ob_scale[0], ob_scale[2], ob_scale[4]], [ob_scale[0], ob_scale[2], ob_scale[5]],
                 [ob_scale[0], ob_scale[3], ob_scale[4]], [ob_scale[0], ob_scale[3], ob_scale[5]],
                 [ob_scale[1], ob_scale[2], ob_scale[4]], [ob_scale[1], ob_scale[2], ob_scale[5]],
                 [ob_scale[1], ob_scale[3], ob_scale[4]], [ob_scale[1], ob_scale[3], ob_scale[5]]]
    
    inipos = [ob[0], ob[1], ob[2]]
    
    ob_vertex = [SetRotationPoint(inipos, ob_vertex[0], rotation), SetRotationPoint(inipos, ob_vertex[1], rotation),
                 SetRotationPoint(inipos, ob_vertex[2], rotation), SetRotationPoint(inipos, ob_vertex[3], rotation),
                 SetRotationPoint(inipos, ob_vertex[4], rotation), SetRotationPoint(inipos, ob_vertex[5], rotation),
                 SetRotationPoint(inipos, ob_vertex[6], rotation), SetRotationPoint(inipos, ob_vertex[7], rotation),]
    
    # find the ob aabb scale after rotation
    ob_scale = []
    obxmin = 1000
    obxmax = 0
    obymin = 1000
    obymax = 0
    obzmin = 1000
    obzmax = 0
    for i in ob_vertex:
        if i[0] <= obxmin: obxmin = i[0]
        if i[0] > obxmax: obxmax = i[0]
        if i[1] <= obymin: obymin = i[1]
        if i[1] > obymax: obymax = i[1]
        if i[2] <= obzmin: obzmin = i[2]
        if i[2] > obzmax: obzmax = i[2]
    ob_scale = [obxmin, obxmax, obymin, obymax, obzmin, obzmax]

    return ob_scale, ob_vertex

def quat_mul_NotForTensor(a, b):    
    '''   rotate a then b   '''
    x1, y1, z1, w1 = a.x, a.y, a.z, a.w
    x2, y2, z2, w2 = b.x, b.y, b.z, b.w
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = gymapi.Quat(x, y, z, w)

    return quat

def H_2_Transform(H):
    so3r3_repr = SO3_R3(R=H[:3, :3].view(1,3,3), t=H[:3, -1].view(1,3))

    p = H[:3,-1]
    q = so3r3_repr.to_quaternion()[0,...]

    p = gymapi.Vec3(x=p[0], y=p[1], z=p[2])
    q = gymapi.Quat(w=q[0], x=q[1], y=q[2], z=q[3])

    return gymapi.Transform(p, q)

def Transform_2_H(T):
    H = torch.eye(4)
    # set position
    H[0, -1] = T.p.x
    H[1, -1] = T.p.y
    H[2, -1] = T.p.z
    # set rotation
    q = torch.Tensor([T.r.w, T.r.x, T.r.y, T.r.z])
    so3_repr = th.geometry.SO3(quaternion=q).to_matrix()
    H[:3,:3] = so3_repr
    return H

def euler_xyz_to_matrix(x=0, y=0, z=0):
    deg = torch.tensor([x, y, z])
    mat = torch.eye(4)
    mat[:3, :3] = euler_angles_to_matrix(deg, "XYZ")
    return mat
    
def euler_angle_to_quaternion(rad): # rad: tensor
    q = matrix_to_quaternion(euler_angles_to_matrix(rad, "XYZ"))
    return q

def pq_to_H(p, q):
    # expects as input: quaternion with convention [x y z w]
    # arrange quaternion with convention [w x y z] for theseus
    q = torch.Tensor([q[3], q[0], q[1], q[2]])
    so3_repr = th.geometry.SO3(quaternion=q).to_matrix()
    H = torch.eye(4).to(p)
    H[:3,:3] = so3_repr
    H[:3, -1] = p
    return H
    