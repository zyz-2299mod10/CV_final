import torch
import numpy as np
import open3d as o3d

def compute_camera_intrinsics_matrix(image_heigth, image_width, horizontal_fov):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    # K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda:0')
    return K

def world_2_camera_frame(cam_coordinate, query):
    '''
    cam_coordinate: camera local coordinate in world
    query: query location in world (tensor)
    '''
    query = np.append(query.numpy(), [1], axis = 0).reshape(4, 1)

    world2cam = np.linalg.inv(cam_coordinate)
    query2cam = world2cam @ query
    
    return query2cam[:3].reshape(3, )
