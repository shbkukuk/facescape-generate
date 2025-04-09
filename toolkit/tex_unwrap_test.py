import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
from src.renderer import render_cvcam
import timeit
import csv
import numpy as np, cv2, trimesh
from src.facescape_fitter import facescape_fitter
from src.renderer import render_orthcam
from src.renderer import render_cvcam

np.random.seed(1000)

# Initialize model and fitter
fs_fitter = facescape_fitter(fs_file=r"C:/Users/SBK/Desktop/master-thesis/code/facescape-generate/toolkit/facescape_bilinear_model_v1_6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend='dlib')  # or 'face_alignment'

# Fit id to image
src_path = "./test_data/tu_color.jpg"
src_img = cv2.imread(src_path)
print(src_img.shape)
assert src_img.shape[2] == 3
kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)  # fit model
id, _, scale, trans, rot_vector = params

# # Get texture
# texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
# filename = './demo_output/kardelen/kardicv.jpg'
# cv2.imwrite(filename, texture)

# Save base mesh
mesh.export(output_dir='./demo_output', file_name='tu_color_mesh', texture_name='tu_color_mesh.jpg', enable_vc=False, enable_vt=True)

# Rescale x-axis of vertices
def rescale_vertices(mesh, img_width, img_height):
    for vertex in mesh.vertices:
        vertex[1] = vertex[1] * 0.01
        vertex[0] = vertex[0] * 0.01
        vertex[2] = vertex[2] * 0.01
    return mesh

# Load the mesh
mesh = trimesh.load('./demo_output/tu_color.obj')

# Rescale vertices
img_height, img_width = src_img.shape[:2]
mesh = rescale_vertices(mesh, img_width, img_height)

# Save the rescaled mesh
mesh.export('./demo_output/tu_color_rescaled_wh.obj')
