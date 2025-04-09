import numpy as np
import cv2
from src.facescape_fitter import facescape_fitter
import trimesh
from src.renderer import render_orthcam, render_glcam
import os

def normalize_depth_map(depth_map):
    """
    Normalize depth map to range [0, 1] and invert if needed
    (assuming darker values in depth map represent deeper points)
    """
    # Convert to float32 and normalize to [0, 1]
    if isinstance(depth_map, str):
        # If input is a path
        depth = cv2.imread(depth_map, cv2.IMREAD_GRAYSCALE)
    else:
        # If input is an array
        depth = depth_map.astype(np.float32)
        if len(depth.shape) == 3:  # If RGB/BGR, convert to grayscale
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    # Ensure we're working with float32
    depth = depth.astype(np.float32)
    
    # Normalize to [0, 1]
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    
    # Invert if needed (make closer points have higher values)
    if np.mean(depth[depth > 0.9]) > np.mean(depth[depth < 0.1]):
        depth = 1 - depth
        
    return depth

def extract_face_texture(rgb_image, mesh, mesh_verts_img, fs_fitter):
    """
    Extract texture from RGB image for the visible parts of the face.
    
    Args:
        rgb_image: Input RGB image
        mesh: 3D mesh
        mesh_verts_img: 2D projected vertices
        fs_fitter: FaceScape fitter instance
    """
    h, w = rgb_image.shape[:2]
    texture = np.zeros((4096, 4096, 3), dtype=np.uint8)
    
    # Create visibility mask using depth rendering
    depth_map, _ = render_glcam(mesh, rend_size=(h, w))
    visibility_mask = depth_map > 0
    
    # Extract texture using UV coordinates
    for face in mesh.faces:
        face_vertices, face_normals, tc, material = face
        
        # Skip faces with large UV coordinate differences
        if max(abs(fs_fitter.texcoords[tc[0] - 1][0] - fs_fitter.texcoords[tc[1] - 1][0]),
               abs(fs_fitter.texcoords[tc[0] - 1][0] - fs_fitter.texcoords[tc[2] - 1][0]),
               abs(fs_fitter.texcoords[tc[1] - 1][0] - fs_fitter.texcoords[tc[2] - 1][0]),
               abs(fs_fitter.texcoords[tc[0] - 1][1] - fs_fitter.texcoords[tc[1] - 1][1]),
               abs(fs_fitter.texcoords[tc[0] - 1][1] - fs_fitter.texcoords[tc[2] - 1][1]),
               abs(fs_fitter.texcoords[tc[1] - 1][1] - fs_fitter.texcoords[tc[2] - 1][1])) > 0.3:
            continue
            
        # Get triangle vertices in image space
        tri1 = np.float32([[[(h - int(mesh_verts_img[face_vertices[0] - 1, 1])),
                            int(mesh_verts_img[face_vertices[0] - 1, 0])],
                           [(h - int(mesh_verts_img[face_vertices[1] - 1, 1])),
                            int(mesh_verts_img[face_vertices[1] - 1, 0])],
                           [(h - int(mesh_verts_img[face_vertices[2] - 1, 1])),
                            int(mesh_verts_img[face_vertices[2] - 1, 0])]]])
        
        # Get triangle vertices in texture space
        tri2 = np.float32([[[4096 - fs_fitter.texcoords[tc[0] - 1][1] * 4096,
                            fs_fitter.texcoords[tc[0] - 1][0] * 4096],
                           [4096 - fs_fitter.texcoords[tc[1] - 1][1] * 4096,
                            fs_fitter.texcoords[tc[1] - 1][0] * 4096],
                           [4096 - fs_fitter.texcoords[tc[2] - 1][1] * 4096,
                            fs_fitter.texcoords[tc[2] - 1][0] * 4096]]])
        
        # Get bounding rectangles
        r1 = cv2.boundingRect(tri1)
        r2 = cv2.boundingRect(tri2)
        
        # Prepare triangles for warping
        tri1Cropped = []
        tri2Cropped = []
        for i in range(3):
            tri1Cropped.append(((tri1[0][i][1] - r1[1]), (tri1[0][i][0] - r1[0])))
            tri2Cropped.append(((tri2[0][i][1] - r2[1]), (tri2[0][i][0] - r2[0])))
            
        # Apply warp to small rectangular patches
        img1Cropped = rgb_image[r1[0]:r1[0] + r1[2], r1[1]:r1[1] + r1[3]]
        warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
        
        # Create and apply mask
        mask = np.zeros((r2[2], r2[3], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)
        
        # Warp and blend
        img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[3], r2[2]), 
                                    None, flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT_101)
        img2Cropped = img2Cropped * mask
        
        # Copy to texture map
        texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = \
            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] * ((1.0, 1.0, 1.0) - mask)
        texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] += img2Cropped.astype(np.uint8)
    
    return texture

def create_complete_head_mesh(rgb_image, depth_map, model_path):
    """
    Create a complete 3D head mesh including the back of the head.
    
    Args:
        rgb_image: RGB image as numpy array or path
        depth_map: Depth map as numpy array or path
        model_path: Path to the FaceScape model file
    """
    # Initialize the FaceScape fitter with full head model
    fs_fitter = facescape_fitter(fs_file=model_path, kp2d_backend='dlib')
    
    # Load RGB image if path is provided
    if isinstance(rgb_image, str):
        rgb_image = cv2.imread(rgb_image)
    
    # Ensure input image is RGB
    assert rgb_image is not None and rgb_image.shape[2] == 3, "Input image must be RGB"
    
    # Normalize depth map
    depth = normalize_depth_map(depth_map)
    
    # Scale depth values to reasonable 3D range (e.g., -0.1 to 0.1 meters)
    depth = (depth * 0.2) - 0.1
    
    # Detect 2D facial landmarks
    kp2d = fs_fitter.detect_kp2d(rgb_image)
    if len(kp2d) == 0:
        raise ValueError("No face detected in the image")
    
    # Convert 2D landmarks to 3D using depth information
    kp3d = []
    for x, y in kp2d:
        x, y = int(x), int(y)
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            z = depth[y, x]
            kp3d.append([x, y, z])
        else:
            # If landmark is outside depth map bounds, use average depth
            z = np.mean(depth[depth > 0])
            kp3d.append([x, y, z])
    
    kp3d = np.array(kp3d)
    
    # Fit the 3D model using the 3D landmarks
    mesh, params = fs_fitter.fit_kp3d(kp3d)
    id_params, exp_params, scale, trans, rot_vector = params
    
    # Generate complete head mesh using the fitted parameters
    complete_mesh = fs_fitter.gen_full(id_params, exp_params)
    
    # Apply the same transformation to the complete mesh as the face mesh
    complete_mesh.vertices = complete_mesh.vertices * scale
    
    # Get texture from the input image for the visible parts
    _, _, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)
    texture = extract_face_texture(rgb_image, mesh, mesh_verts_img, fs_fitter)
    
    return complete_mesh, texture, params

def main():
    # Example usage
    rgb_path = "./test_data/tu_color.jpg"
    depth_path = "./test_data/tu_depth.jpg"
    # Use the full head model instead of front-only model
    model_path = "./facescape_bilinear_model_v1_6/facescape_bm_v1.6_847_50_52_id.npz"
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Create complete head mesh directly from paths
    mesh, texture, params = create_complete_head_mesh(rgb_path, depth_path, model_path)
    
    # Save the results
    mesh.export(output_dir='./output', file_name='complete_head_mesh', 
                texture_name='texture.jpg', enable_vc=False, enable_vt=True)
    
    # Save texture maps
    cv2.imwrite('./output/texture.jpg', texture)
    
    # Save visualization of the textured mesh from different angles
    angles = [0, 45, 90, -45, -90]  # Different viewing angles
    for angle in angles:
        # Render the textured mesh
        _, color = render_glcam(mesh, 
                              Rt=np.array([[np.cos(np.radians(angle)), 0, np.sin(np.radians(angle)), 0],
                                         [0, 1, 0, 0],
                                         [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle)), 0]]),
                              rend_size=(512, 512))
        cv2.imwrite(f'./output/view_{angle}.jpg', color)

if __name__ == "__main__":
    main() 