import open3d as o3d
import copy

class icp():
    def __init__(self, source, target, voxel_size = 10):
        '''
        args:
            source: open3d.pcd
            target: open3d.pcd
        '''
        self.source = source
        self.target = target
        self.voxel_size = voxel_size
    
    def draw_registration_result(self, source, target, store_name, transform = None):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 0])

        if transform is not None:
            source_temp.transform(transform)
        
        visualize_pcd = source_temp + target_temp
        o3d.io.write_point_cloud(store_name, visualize_pcd)
    
    def preprocess_point_cloud(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def get_init_trans(self, visualize = False):        
        self.source_down, source_fpfh = self.preprocess_point_cloud(self.source, self.voxel_size)
        self.target_down, target_fpfh = self.preprocess_point_cloud(self.target, self.voxel_size/2)

        result_ransac = self.execute_global_registration(self.source_down, self.target_down,
                                            source_fpfh, target_fpfh, self.voxel_size)

        if visualize:
            store_name = './visualize_result/close/first_align_pcd.ply'
            self.draw_registration_result(self.source_down, self.target_down, store_name, result_ransac.transformation)
        
        return result_ransac.transformation

    def icp_refinement(self, transformation, visualize = False):
        distance_threshold = self.voxel_size * 0.4
        self.target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            self.source, self.target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

        if visualize:
            store_name = './visualize_result/close/icp_align_pcd.ply'
            self.draw_registration_result(self.source, self.target, store_name, result.transformation)

        return result.transformation