import os
import glob
import json
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

def get_alignment_transform(room_txt_path):
    if not os.path.exists(room_txt_path):
        return np.identity(4)
    try:
        transform = np.loadtxt(room_txt_path)
        if transform.shape == (4, 4):
            return transform
        else:
            return np.identity(4)
    except Exception:
        return np.identity(4)

def load_alignment_angles(area_alignment_file_path):
    angles = {}
    if not os.path.exists(area_alignment_file_path):
        return angles
    with open(area_alignment_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                room_name, angle = parts
                try:
                    angles[room_name] = float(angle)
                except ValueError:
                    pass
    return angles

def create_z_rotation_matrix(angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    t = np.identity(4)
    t[0, 0] = cos_a
    t[0, 1] = -sin_a
    t[1, 0] = sin_a
    t[1, 1] = cos_a
    return t

def project_point_to_image(points_world, pose_path, depth_path, color_path, image_size_hw):
    try:
        with open(pose_path, 'r') as fin:
            pose_data = json.load(fin)
        depth_intrinsic = np.array(pose_data['camera_k_matrix'])

        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]

        depth_img_raw = cv2.imread(depth_path, -1)
        if depth_img_raw is None:
            return np.array([], dtype=int), None, np.array([]), (0,0,0)
        
        depth_shift = 512.0
        depth = depth_img_raw / depth_shift
        depth_mask = (depth_img_raw != 0) & (depth_img_raw != (2**16 - 1))

        color_image = cv2.imread(color_path)
        if color_image is None:
            return np.array([], dtype=int), None, np.array([]), (0,0,0)
        color_image_shape = color_image.shape

        pose_rt_matrix = pose_data['camera_rt_matrix']
        pose = np.array(pose_rt_matrix)
        full_pose = np.vstack([pose, np.array([0, 0, 0, 1])])

        points_h = np.hstack((points_world[:, :3], np.ones((points_world.shape[0], 1))))
        points_camera_frame_h = np.dot(points_h, full_pose.T)
        points_camera_frame = points_camera_frame_h[:, :3]

        z_cam = points_camera_frame[..., 2]
        valid_z_mask = z_cam > 1e-6

        u = np.zeros_like(z_cam)
        v = np.zeros_like(z_cam)

        u[valid_z_mask] = (points_camera_frame[valid_z_mask, 0] * fx / z_cam[valid_z_mask]) + cx
        v[valid_z_mask] = (points_camera_frame[valid_z_mask, 1] * fy / z_cam[valid_z_mask]) + cy
        d_cam_coords = z_cam

        u_int = (u + 0.5).astype(np.int32)
        v_int = (v + 0.5).astype(np.int32)
        
        img_h, img_w = image_size_hw[0], image_size_hw[1]

        point_valid_mask = valid_z_mask & \
                           (d_cam_coords >= 0) & \
                           (u_int < img_w) & \
                           (v_int < img_h) & \
                           (u_int >= 0) & \
                           (v_int >= 0)
        
        point_valid_indices_initial = np.where(point_valid_mask)[0]
        if len(point_valid_indices_initial) == 0:
            return np.array([], dtype=int), None, np.array([]), color_image_shape

        valid_u = u_int[point_valid_indices_initial]
        valid_v = v_int[point_valid_indices_initial]
        valid_d_camera = d_cam_coords[point_valid_indices_initial]

        image_depth_at_pixels = depth[valid_v, valid_u]
        depth_map_mask_at_pixels = depth_mask[valid_v, valid_u]

        depth_consistent_mask = depth_map_mask_at_pixels & \
                                (np.abs(image_depth_at_pixels - valid_d_camera) <= 0.2 * image_depth_at_pixels) & \
                                (image_depth_at_pixels > 1e-6)

        final_valid_point_indices_in_original_input = point_valid_indices_initial[depth_consistent_mask]
        
        if len(final_valid_point_indices_in_original_input) == 0:
            return np.array([], dtype=int), None, np.array([]), color_image_shape

        final_u = valid_u[depth_consistent_mask]
        final_v = valid_v[depth_consistent_mask]

        point2image_coords_2d = np.vstack([final_u, final_v]).T
        
        return final_valid_point_indices_in_original_input, None, point2image_coords_2d, color_image_shape

    except Exception:
        return np.array([], dtype=int), None, np.array([]), (0,0,0)

def parse_s3dis_scene_name(pth_file_basename, area_dirname):
    room_identifier = pth_file_basename.replace('.pth', '')
    s2d3d_area_name = area_dirname.lower()
    return s2d3d_area_name, room_identifier

def main(args):
    s3dis_processed_root = args.s3dis_processed_root
    s3dis_non_aligned_root = args.s3dis_non_aligned_root
    s2d3d_root = args.s2d3d_root
    output_idx_numpy_root = args.output_idx_numpy_root
    
    processing_image_size_hw = tuple(map(int, args.image_size_hw.split(','))) 
    img_h, img_w = processing_image_size_hw[0], processing_image_size_hw[1]

    s3dis_area_dirs = sorted([d for d in os.listdir(s3dis_processed_root) if d.startswith('Area_') and os.path.isdir(os.path.join(s3dis_processed_root, d))])

    for area_dirname in tqdm(s3dis_area_dirs, desc="Processing Areas"):
        alignment_angle_file = os.path.join(s3dis_non_aligned_root, area_dirname, f'{area_dirname}_alignmentAngle.txt')
        room_to_angle_map = load_alignment_angles(alignment_angle_file)
        
        area_path = os.path.join(s3dis_processed_root, area_dirname)
        pth_files = sorted(glob.glob(os.path.join(area_path, '*.pth')))

        if not pth_files:
            continue

        for pth_file_path in tqdm(pth_files, desc=f"Scenes in {area_dirname}", leave=False):
            try:
                scene_data = torch.load(pth_file_path)
                if isinstance(scene_data, dict) and 'coords' in scene_data : 
                     points_world_s3dis = scene_data['coords'] 
                elif isinstance(scene_data, np.ndarray): 
                    points_world_s3dis = scene_data[:, :3].astype(np.float32)
                else:
                    continue
                
                if points_world_s3dis.shape[0] == 0:
                    continue

                room_identifier_for_lookup = os.path.basename(pth_file_path).replace('.pth', '')
                
                alignment_angle = room_to_angle_map.get(room_identifier_for_lookup)

                if alignment_angle is not None and alignment_angle != 0:
                    centroid = points_world_s3dis.mean(0)
                    t_rot = create_z_rotation_matrix(alignment_angle)
                    points_h = np.hstack((points_world_s3dis - centroid, np.ones((points_world_s3dis.shape[0], 1))))
                    points_to_project_h = (t_rot @ points_h.T).T
                    points_to_project = points_to_project_h[:, :3] + centroid
                else:
                    points_to_project = points_world_s3dis

                pth_file_basename = os.path.basename(pth_file_path)
                s2d3d_area_name, room_identifier = parse_s3dis_scene_name(pth_file_basename, area_dirname)
                
                pose_dir = os.path.join(s2d3d_root, s2d3d_area_name, 'data', 'pose')
                depth_dir = os.path.join(s2d3d_root, s2d3d_area_name, 'data', 'depth')
                rgb_dir = os.path.join(s2d3d_root, s2d3d_area_name, 'data', 'rgb')

                pose_files = sorted(glob.glob(os.path.join(pose_dir, f'*{room_identifier}_*_pose.json')))
                
                if not pose_files:
                    continue

                for pose_path in tqdm(pose_files, desc="Images", leave=False):
                    try:
                        pose_basename = os.path.basename(pose_path)
                        img_identifier = pose_basename.replace('_pose.json', '')

                        depth_path = os.path.join(depth_dir, img_identifier + '_depth.png')
                        rgb_path = os.path.join(rgb_dir, img_identifier + '_rgb.png')
                        
                        if not (os.path.exists(depth_path) and os.path.exists(rgb_path)):
                            continue
                        
                        pixel_to_s3dis_idx_map = np.full((img_h, img_w), -1, dtype=np.int32)

                        valid_indices, _, valid_coords_2d, _ = \
                            project_point_to_image(points_to_project, pose_path, depth_path, rgb_path, processing_image_size_hw)

                        if valid_indices.size > 0:
                            for i in range(len(valid_indices)):
                                s3dis_point_original_idx = valid_indices[i]
                                pixel_u = int(valid_coords_2d[i, 0]) 
                                pixel_v = int(valid_coords_2d[i, 1])
                                
                                if 0 <= pixel_v < img_h and 0 <= pixel_u < img_w:
                                    pixel_to_s3dis_idx_map[pixel_v, pixel_u] = s3dis_point_original_idx
                        
                        s2d3d_rgb_image_rel_path = os.path.relpath(rgb_path, s2d3d_root)
                        
                        base_output_dir_for_rgb_type = os.path.dirname(s2d3d_rgb_image_rel_path) 
                        modified_output_dir_for_rgb_type = base_output_dir_for_rgb_type + "_s3dis_idx_mapping" 
                        
                        output_npy_dir = os.path.join(output_idx_numpy_root, modified_output_dir_for_rgb_type)
                        os.makedirs(output_npy_dir, exist_ok=True)
                        
                        npy_basename = os.path.basename(s2d3d_rgb_image_rel_path).replace('.png', '.npy')
                        output_npy_path = os.path.join(output_npy_dir, npy_basename)
                        
                        np.save(output_npy_path, pixel_to_s3dis_idx_map)
                    except Exception:
                        continue
            except Exception:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3dis_processed_root', type=str, required=True)
    parser.add_argument('--s3dis_non_aligned_root', type=str, required=True)
    parser.add_argument('--s2d3d_root', type=str, required=True)
    parser.add_argument('--output_idx_numpy_root', type=str, required=True)
    parser.add_argument('--image_size_hw', type=str, default="1080,1080")
    
    parsed_args = parser.parse_args()
    main(parsed_args)