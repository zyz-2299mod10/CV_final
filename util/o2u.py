from argparse import ArgumentParser
import os

def generate_obj_urdf(obj_path):
    '''
        creates a urdf object on the fly which is needed as objects can only be spawned from a urdf file
        returns the path to the urdf file
    '''
    obj_path = obj_path[:-4]
    excl_filename = obj_path.split('/')[-1]
    obj_path = obj_path.replace(excl_filename, "")
    print(obj_path)
    
    urdf_path = os.path.join(obj_path, "urdf/")
    with open(str(urdf_path) + excl_filename + '.urdf', 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<robot name="obj.urdf">\n')
        f.write('  <link name="baseLink">\n')
        f.write('    <visual>\n')
        f.write('      <origin rpy="0 0 0" xyz="0 0 0"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename="' + obj_path + str(excl_filename) + '.obj" />\n')
        f.write('      </geometry>\n')
        f.write('    </visual>\n')
        f.write('    <collision>\n')
        f.write('      <origin rpy="0 0 0" xyz="0 0 0"/>\n')
        f.write('      <geometry>\n')
        f.write('        <mesh filename="' + obj_path + str(excl_filename) + '.obj" />\n')
        f.write('      </geometry>\n')
        f.write('    </collision>\n')
        f.write('  </link>\n')
        f.write('</robot>')
    return str(urdf_path) + '.urdf'

parser = ArgumentParser()
parser.add_argument("-op", help = 'obj path', type = str, dest = "obj_path")

args = parser.parse_args()

if __name__ == "__main__":
    print(args.obj_path)
    generate_obj_urdf(args.obj_path)


