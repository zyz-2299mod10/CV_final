import re
import os

import open3d as o3d
import numpy as np

class get_urdf_info:
    def __init__(self, 
                 root: str,
                 name: str,
                 ):
              
        self.urdf_root = root
        self.usb_place = name
        
        self.filename = os.path.join(self.urdf_root, self.usb_place)
        
        # read urdf file
        with open(self.filename, 'r') as file:
            self.file_content = file.read()           
        
    def get_collision_pathName_scale(self):  
        info = {
                "filename": str,
                "scale": list
                }
        
        # get .obj file name & scale         
        pattern = re.compile(r'<collision>.*?<mesh filename="(.*?)"(?: scale\s*=\s*"(.*?)")?.*?</collision>', re.DOTALL)
        match = pattern.search(self.file_content)

        if match:
            mesh_filename = match.group(1) # mesh_filename: ./obj-and-urdf/*.obj
            scale_values = match.group(2)

            if scale_values:                
                scale_values = np.array(scale_values.split()).astype(float)
                
                info["filename"] = os.path.abspath(mesh_filename)
                info["scale"] = scale_values.tolist()
                
                return info
            else:               
                info["filename"] = os.path.abspath(mesh_filename)              
                info["scale"] = [1, 1, 1]
                
                return info 
                
        else:
            assert "No file."   
    
    def get_mesh_aabb_size(self, error = 0):
        info = self.get_collision_pathName_scale()
        
        #filename = os.path.abspath(info["filename"])
        
        mesh = o3d.io.read_triangle_mesh(info["filename"])
        #print(mesh)
        aabb = mesh.get_axis_aligned_bounding_box()
        aabb_min = aabb.get_min_bound()
        aabb_max = aabb.get_max_bound()
        
        # scale
        scale = info["scale"]
        x_scale = scale[0]
        y_scale = scale[1]
        z_scale = scale[2]
                
        x = (aabb_max[0] - aabb_min[0]) * x_scale + error
        y = (aabb_max[1] - aabb_min[1]) * y_scale + error
        z = (aabb_max[2] - aabb_min[2]) * z_scale + error
        
        return np.array([x, y, z])
        
