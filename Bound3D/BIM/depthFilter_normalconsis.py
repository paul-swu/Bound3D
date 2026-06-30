import cv2
import numpy as np
import os
import random

def detect_normal_edges(normal_img):
    return None


depth_dir = None
rgb_dir = None
normal_dir = None
output_dir_base = None
os.makedirs(output_dir_base, exist_ok=True)
n_images_to_process = 30
random_seed = 42
random.seed(random_seed)
boundary_point_filter_threshold = 100000 
depth_gaussian_kernel_size = (3, 3) 
depth_canny_low_threshold = 20
depth_canny_high_threshold = 40
depth_missing_dilate_iterations = 2 
rgb_gaussian_kernel_size = (3, 3)  
rgb_canny_low_threshold = 85
rgb_canny_high_threshold = 220

search_radius_around_depth_edge = 8 

param_string = (f"d_k{depth_gaussian_kernel_size[0]}_lt{depth_canny_low_threshold}_ht{depth_canny_high_threshold}_dil{depth_missing_dilate_iterations}_"
                f"rgb_k{rgb_gaussian_kernel_size[0]}_lt{rgb_canny_low_threshold}_ht{rgb_canny_high_threshold}_"
                f"searchR{search_radius_around_depth_edge}_seed{random_seed}")
output_run_dir_name = f"depth_refined_rgb_canny_{param_string}"
output_run_dir = os.path.join(output_dir_base, output_run_dir_name)
os.makedirs(output_run_dir, exist_ok=True)

stats_file = os.path.join(output_dir_base, f"stats_{output_run_dir_name}.txt")
with open(stats_file, 'w') as f:
    f.write("Image File,D_K,D_LT,D_HT,D_Dil,RGB_K,RGB_LT,RGB_HT,SearchR,"
            "Original RGB Edge Pts,Final Edge Pts,Total Pixels,Final Pct(%),Status\n")

depth_files = glob.glob(os.path.join(depth_dir, "*.png"))
if not depth_files:
    exit()

if len(depth_files) < n_images_to_process:
    selected_depth_files = depth_files
else:
    selected_depth_files = random.sample(depth_files, n_images_to_process)

processed_count = 0
skipped_count = 0

for i, depth_path in enumerate(selected_depth_files):
    depth_base_name_with_ext = os.path.basename(depth_path)
    depth_base_name = os.path.splitext(depth_base_name_with_ext)[0]

    rgb_filename = depth_base_name_with_ext.replace("_depth.png", "_rgb.png")
    rgb_path = os.path.join(rgb_dir, rgb_filename)

    normal_filename = depth_base_name_with_ext.replace("_depth.png", "_normal.png")
    normal_path = os.path.join(normal_dir, normal_filename)

    depth_image_16bit = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(rgb_path)
    normal_image = cv2.imread(normal_path)

    if depth_image_16bit is None or rgb_image is None or normal_image is None:
        status_msg = "Read Error"
        if depth_image_16bit is None: status_msg += " Depth File"
        if rgb_image is None: status_msg += " RGB File"
        if normal_image is None: status_msg += " Normal File"
        with open(stats_file, 'a') as f:
            f.write(f"{depth_base_name_with_ext},{depth_gaussian_kernel_size[0]},{depth_canny_low_threshold},{depth_canny_high_threshold},{depth_missing_dilate_iterations},"
                    f"{rgb_gaussian_kernel_size[0]},{rgb_canny_low_threshold},{rgb_canny_high_threshold},{search_radius_around_depth_edge},"
                    f"N/A,N/A,N/A,N/A,{status_msg}\n")
        skipped_count += 1
        continue

    missing_mask_depth = (depth_image_16bit == 65535)
    valid_depth_16bit = depth_image_16bit.copy()
    valid_depth_16bit[missing_mask_depth] = 0 

    depth_for_canny_8bit = np.zeros_like(valid_depth_16bit, dtype=np.uint8)
    if np.any(~missing_mask_depth):
        min_val, max_val, _, _ = cv2.minMaxLoc(valid_depth_16bit[~missing_mask_depth])
        if max_val > min_val:
            normalized_valid_area = cv2.normalize(valid_depth_16bit[~missing_mask_depth], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_for_canny_8bit[~missing_mask_depth] = normalized_valid_area.flatten()
        else:
            depth_for_canny_8bit[~missing_mask_depth] = 128
    
    depth_blurred_8bit = cv2.GaussianBlur(depth_for_canny_8bit, depth_gaussian_kernel_size, 0)
    depth_canny_edges_raw = cv2.Canny(depth_blurred_8bit, depth_canny_low_threshold, depth_canny_high_threshold)
    
    kernel_dilate_missing = np.ones((3,3), np.uint8)
    dilated_original_missing_mask = cv2.dilate(missing_mask_depth.astype(np.uint8), kernel_dilate_missing, iterations=depth_missing_dilate_iterations)
    depth_canny_edges_final = depth_canny_edges_raw.copy()
    depth_canny_edges_final[dilated_original_missing_mask > 0] = 0

    dilation_kernel_size = 2 * search_radius_around_depth_edge + 1
    search_mask_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    depth_search_mask = cv2.dilate(depth_canny_edges_final, search_mask_kernel, iterations=1) 

    normal_edges = detect_normal_edges(normal_image)

    gray_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    rgb_blurred = cv2.GaussianBlur(gray_rgb_image, rgb_gaussian_kernel_size, 0)
    rgb_canny_edges_unrefined = cv2.Canny(rgb_blurred, rgb_canny_low_threshold, rgb_canny_high_threshold)
    
    refined_rgb_edges = cv2.bitwise_and(rgb_canny_edges_unrefined, rgb_canny_edges_unrefined, mask=depth_search_mask)

    final_boundary_points_count = np.sum(refined_rgb_edges > 0)
    rgb_canny_unrefined_count = np.sum(rgb_canny_edges_unrefined > 0) 
    total_pixels = refined_rgb_edges.size
    final_boundary_percentage = (final_boundary_points_count / total_pixels) * 100 if total_pixels > 0 else 0
    
    if final_boundary_points_count > boundary_point_filter_threshold:
        status_msg = "Skipped (Too many edge points)"
        skipped_count += 1
    else:
        status_msg = "Processed"
        processed_count += 1

        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_0_original_rgb.png"), rgb_image)
        depth_vis_8bit = cv2.normalize(depth_image_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_1_original_depth_vis.png"), depth_vis_8bit)
        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_2_depth_canny_final.png"), depth_canny_edges_final)
        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_3_depth_search_mask_radius{search_radius_around_depth_edge}.png"), depth_search_mask)
        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_4_rgb_canny_unrefined.png"), rgb_canny_edges_unrefined)
        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_5_refined_rgb_edges.png"), refined_rgb_edges)

        visualization_final = rgb_image.copy()
        green_overlay = np.zeros_like(visualization_final)
        green_overlay[depth_search_mask == 255] = [0, 180, 0] 
        visualization_final = cv2.addWeighted(visualization_final, 1.0, green_overlay, 0.3, 0)
        visualization_final[refined_rgb_edges > 0] = [0, 0, 255] 

        info_l1 = f"D_K{depth_gaussian_kernel_size[0]} LT{depth_canny_low_threshold} HT{depth_canny_high_threshold} R{search_radius_around_depth_edge}"
        info_l2 = f"RGB_K{rgb_gaussian_kernel_size[0]} LT{rgb_canny_low_threshold} HT{rgb_canny_high_threshold}"
        info_l3 = f"Final Pts: {final_boundary_points_count} ({final_boundary_percentage:.2f}%)"
        
        cv2.putText(visualization_final, info_l1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(visualization_final, info_l1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(visualization_final, info_l2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(visualization_final, info_l2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(visualization_final, info_l3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(visualization_final, info_l3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(output_run_dir, f"{depth_base_name}_6_FINAL_visualization.png"), visualization_final)

    with open(stats_file, 'a') as f:
        f.write(f"{depth_base_name_with_ext},{depth_gaussian_kernel_size[0]},{depth_canny_low_threshold},{depth_canny_high_threshold},{depth_missing_dilate_iterations},"
                f"{rgb_gaussian_kernel_size[0]},{rgb_canny_low_threshold},{rgb_canny_high_threshold},{search_radius_around_depth_edge},"
                f"{rgb_canny_unrefined_count},{final_boundary_points_count},{total_pixels},{final_boundary_percentage:.2f},{status_msg}\n")


