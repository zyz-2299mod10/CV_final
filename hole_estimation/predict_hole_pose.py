import os
import torch
import open3d as o3d
import numpy as np
import copy
import importlib
import torch
from scipy.spatial.transform import Rotation as R
from hole_estimation.ICP import icp
from hole_estimation.mankey.models.utils import compute_rotation_matrix_from_ortho6d

class CoarseMover(object):
    def __init__(self, model_path='kpts/2022-02-??_??-??', model_name='pointnet2_kpts', checkpoint_name='best_model_e_?.pth', use_cpu=False, out_channel=9):
        '''MODEL LOADING'''
        exp_dir = './hole_estimation/mankey/log/' +  model_path
        model = importlib.import_module('hole_estimation.mankey.models.' + model_name)
        self.network = model.get_model()
        self.network.apply(self.inplace_relu)
        self.use_cpu = use_cpu
        if not self.use_cpu:
            self.network = self.network.cuda()
        try:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', checkpoint_name)
            checkpoint = torch.load(checkpoint_path)
            # print(checkpoint['epoch'])
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print('Loading model successfully !')
        except:
            print('No existing model...')
            assert False

    def inplace_relu(self, m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True
    
    def crop_point_cloud(self, pcd, mode):
        crop_concat_xyz_in_world = []
        if mode == 'far':
            for idx, xyz in enumerate(pcd):
                x = xyz[0] / 1000  # unit:m
                y = xyz[1] / 1000  # unit:m
                z = xyz[2] / 1000  # unit:m
                if x >= 0.3 and x <= 0.7 and \
                    y >= -0.25 and y <= 0.05 and \
                    z >= 0.35 and z <= 0.45:
                    crop_concat_xyz_in_world.append(xyz)

        elif mode == 'close':
            for idx, xyz in enumerate(pcd):
                x = xyz[0] / 1000  # unit:m
                y = xyz[1] / 1000  # unit:m
                z = xyz[2] / 1000  # unit:m
                if z >= 0.35 and z <= 0.46:
                    crop_concat_xyz_in_world.append(xyz)
        
        pcd = np.array(crop_concat_xyz_in_world)

        return pcd

    def depth_2_pcd(self, depth, factor, K, view_matrix):
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
        xyz = np.concatenate((pt0, pt1, pt2), axis=1)

        xyz_in_world_list = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        down_pcd = pcd.uniform_down_sample(every_k_points=6)
        down_xyz = np.asarray(down_pcd.points)
        down_xyz_in_camera = down_xyz[:10000, :]

        down_xyz_in_world = []
        for xyz in down_xyz_in_camera: # turn pcd into world coordinate
            camera2world = np.array(view_matrix)
            xyz = np.append(xyz, [1], axis=0).reshape(4, 1)
            xyz_world = camera2world.dot(xyz)
            xyz_world = xyz_world[:3] * 1000
            down_xyz_in_world.append(xyz_world)
        xyz_in_world_list.append(down_xyz_in_world)
        concat_xyz_in_world = np.array(xyz_in_world_list)
        concat_xyz_in_world = concat_xyz_in_world.reshape(-1,3)

        return concat_xyz_in_world, choose
    
    def normalize_pcd(self, pcd):
        normal_pcd = copy.deepcopy(pcd)
        c = np.mean(normal_pcd, axis=0)
        normal_pcd = normal_pcd - c
        m = np.max(np.sqrt(np.sum(normal_pcd ** 2, axis=1)))
        normal_pcd = normal_pcd / m

        return normal_pcd, c, m

    def get_hole_pose_from_pcd(self, pcd, mode, use_offset = False, visualize = False):
        '''
        input param: pcd numpy
        return: hole pose matrix
        '''
        if mode == 'far':
            ori_pcd = pcd
            pcd = self.crop_point_cloud(pcd=pcd, mode='far')
        if mode == 'close':
            pcd = self.crop_point_cloud(pcd=pcd, mode='close')

        normal_pcd, centroid, m = self.normalize_pcd(pcd)

        points = np.expand_dims(normal_pcd, axis=0)
        points = torch.Tensor(points)
        points = points.transpose(2, 1) # to Tensor (1*C*N)

        self.network = self.network.eval()
        with torch.no_grad():
            if not self.use_cpu:
                points = points.cuda()
            kpt_of_pred, trans_of_pred, rot_of_pred, mean_kpt_pred, mean_kpt_x_pred, mean_kpt_y_pred, rot_mat_pred, confidence = self.network.forward_test(points)
            mean_kpt_pred = mean_kpt_pred[0].cpu().numpy()
            mean_kpt_x_pred = mean_kpt_x_pred[0].cpu().numpy()
            mean_kpt_y_pred = mean_kpt_y_pred[0].cpu().numpy()
            rot_mat_pred = rot_mat_pred[0].cpu().numpy()
            confidence = confidence[0].cpu().numpy()

            trans_of_pred = trans_of_pred[0].cpu().numpy()
            self.real_kpt_pred = (mean_kpt_pred * m) + centroid  # unit:mm
            self.real_kpt_pred = self.real_kpt_pred.reshape(3, )
            self.real_kpt_x_pred = (mean_kpt_x_pred * m) + centroid  # unit:mm
            self.real_kpt_x_pred = self.real_kpt_x_pred.reshape(1, 3)
            self.real_kpt_y_pred = (mean_kpt_y_pred * m) + centroid  # unit:mm
            self.real_kpt_y_pred = self.real_kpt_y_pred.reshape(1, 3)

            H = np.eye(4)
            H[:3, :3] = rot_mat_pred
            H[:3, 3] = self.real_kpt_pred / 1000 # unit:m

            if visualize:
                points = points.cpu().transpose(2, 1).squeeze().numpy()
                points = (points * m) + centroid

                draw_y = self.real_kpt_y_pred.reshape(3, ) -  H[:3, 1] * 2
                draw_y = draw_y.reshape(1, 3)
                self.visualize_result(points, confidence, [self.real_kpt_pred, self.real_kpt_x_pred, draw_y], mode=mode)

                if mode == 'far':
                    self.visualize_result(ori_pcd, confidence, [self.real_kpt_pred, self.real_kpt_x_pred, self.real_kpt_y_pred], mode=mode)
            
            return H

    def fine_grain_pose(self, pcd, H, visualize = False):
        '''
        Use ICP (iterative closest points) to fine-grain hole pose (only fine-grain yaw)

        arg:
            pcd: unit mm
            init_pose: numpy (4*4)
        '''
        init_pose = copy.deepcopy(H)
        init_pose[:3, -1] = np.array([0, 0, 0])
        
        hole_pcd = self.read_hole_file() # unit: m

        crop_pcd = self.crop_point_cloud(pcd=pcd, mode='close') # crop usb
        c = self.real_kpt_pred # rotation axis
        pcd_in_origin = (crop_pcd - c) / 1000 # unit: m

        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(pcd_in_origin)
        hole_pcd.transform(init_pose)

        # ICP
        finer = icp(hole_pcd, _pcd, 0.005)

        store_name = './visualize_result/close/init_align_pcd.ply'
        finer.draw_registration_result(hole_pcd, _pcd, store_name)

        init_trans = finer.get_init_trans(visualize=visualize)
        icp_trans = finer.icp_refinement(init_trans, visualize=visualize) # result not good

        # transform kpts
        fine_cen = self.real_kpt_pred - c
        fine_cen = np.append(fine_cen, [1], axis = 0).reshape(4, 1)
        fine_cen = init_trans.dot(fine_cen)[:3]
        fine_cen = fine_cen.reshape(3, ) + c

        fine_y = self.real_kpt_y_pred.reshape(3, ) - c
        fine_y = np.append(fine_y, [1], axis = 0).reshape(4, 1)
        fine_y = init_trans.dot(fine_y)[:3]
        fine_y = fine_y.reshape(3, ) + c

        fine_x = self.real_kpt_x_pred.reshape(3, ) - c
        fine_x = np.append(fine_x, [1], axis = 0).reshape(4, 1)
        fine_x = init_trans.dot(fine_x)[:3]
        fine_x = fine_x.reshape(3, ) + c
        if fine_x[2] > fine_cen[2]: # avoid x is above central kpt
            dz = np.abs(fine_x[2] - fine_cen[2])
            fine_x[2] = fine_x[2] - 2 * dz

        # calculate new H
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_cpu = False if device == 'cuda' else True
        vy = fine_cen - fine_y
        vx = fine_cen - fine_x
        vt = np.append(vx, vy, axis = 0)
        vt = np.expand_dims(vt, axis = 0)
        vt = torch.tensor(vt, device = device)
        new_rot = compute_rotation_matrix_from_ortho6d(vt, use_cpu=use_cpu)
        new_rot = new_rot[0].cpu().numpy()
        
        new_H = np.eye(4)
        new_H[:3, :3] = new_rot
        new_H[:3, 3] = self.real_kpt_pred / 1000 # unit:m

        if visualize:
            visualize_pcd = o3d.geometry.PointCloud()
            fine_cen = fine_cen.reshape(1, 3)
            fine_y = fine_y - new_H[:3, 1] * 2
            fine_y = fine_y.reshape(1, 3)
            fine_x = fine_x.reshape(1, 3)
            visualize_pcd.points = o3d.utility.Vector3dVector(np.concatenate((crop_pcd, fine_cen, fine_x, fine_y), axis=0))
            point_color = np.repeat([[0.9, 0.9, 0.9]], crop_pcd.shape[0], axis=0)
            kpt_color = np.array([[0,0,1],[1,0,0],[0,1,0]])
            visualize_pcd.colors = o3d.utility.Vector3dVector(np.concatenate((point_color, kpt_color), axis=0))
            o3d.io.write_point_cloud('./visualize_result/close/fine_kpts.ply', visualize_pcd)
        
        return new_H
    
    def read_hole_file(self, ):
        '''
        read hole obj, then output point cloud (numpy)
        '''
        mesh = o3d.io.read_triangle_mesh('/home/hcis/YongZhe/obj-and-urdf/usb_place_my.obj')
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=450)

        return pcd
    
    def visualize_result(self, pcd, confidence, kpts:list, mode):
        '''
        arg:
            pcd: numpy, unit -> mm
            kpts: [central, x, y] kpt
        '''
        os.makedirs('./visualize_result', exist_ok=True)

        real_kpt_pred = kpts[0]
        real_kpt_x_pred = kpts[1]
        real_kpt_y_pred = kpts[2]

        if mode == 'far':
            front_name = 'before_crop_'
        else:
            front_name = ''

        # visualize heatmap        
        origin_real_pcd = pcd
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(origin_real_pcd)
        heatmap_color = np.repeat(confidence, 3, axis=1).reshape(-1, 3)  # n x 3
        visualize_pcd.colors = o3d.utility.Vector3dVector(heatmap_color)
        o3d.io.write_point_cloud(f'./visualize_result/{mode}/{front_name}visualize_heatmap.ply', visualize_pcd)

        # visualize origin pcd
        visualize_pcd = o3d.geometry.PointCloud()
        visualize_pcd.points = o3d.utility.Vector3dVector(origin_real_pcd)
        o3d.io.write_point_cloud(f'./visualize_result/{mode}/{front_name}visualize_origin.ply', visualize_pcd)

        # visualize kpts
        visualize_pcd = o3d.geometry.PointCloud()
        real_kpt_pred = real_kpt_pred.reshape(1, 3)
        visualize_pcd.points = o3d.utility.Vector3dVector(np.concatenate((origin_real_pcd, real_kpt_pred, real_kpt_x_pred, real_kpt_y_pred), axis=0))
        real_kpt_pred = real_kpt_pred.reshape(3, )
        # kpt_color = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        point_color = np.repeat([[0.9, 0.9, 0.9]], origin_real_pcd.shape[0], axis=0)
        kpt_color = np.array([[0,0,1],[1,0,0],[0,1,0]])

        if mode == 'far':
            kpt_color = np.array([[1,0,0],[1,1,1],[1,1,1]])

        visualize_pcd.colors = o3d.utility.Vector3dVector(np.concatenate((point_color, kpt_color), axis=0))
        o3d.io.write_point_cloud(f'./visualize_result/{mode}/{front_name}visualize_kpts.ply', visualize_pcd)
    
    def predict_kpts_pose(self, depth, factor, K, view_matrix, mode, visualize = False):
        '''
        input: depth (m), np.array
        '''
        pcd, choose = self.depth_2_pcd(depth, factor, K, view_matrix) # pcd unit: mm 
        H = self.get_hole_pose_from_pcd(pcd = pcd, visualize = visualize, mode = mode)
        H[:3, :3] = H[:3, :3] @ R.from_euler("XYZ", np.array([0, 0.5 * np.pi, 0])).as_matrix() # make z-axis up (return from CFVS setting)

        if mode=='close':
            H = self.fine_grain_pose(pcd, H, visualize=visualize)
            H[:3, :3] = H[:3, :3] @ R.from_euler("XYZ", np.array([0, 0.5 * np.pi, 0])).as_matrix() # make z-axis up (return from CFVS setting)


        return H