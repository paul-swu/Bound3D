import os
import torch
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
import random

# =========================================================
# --- V3 Configuration 
# =========================================================

INPUT_DATA_DIR = None

SAMPLING_FILE_PATH = None

OUTPUT_PSEUDO_LABEL_FILE = None

# Algorithm Hyperparameters

MAX_GT_BD_DISTANCE = 2

CHECK_POINT_INTERVAL = 0.1

DENSITY_CHECK_RADIUS = 0.03

MIN_DENSITY_POINTS = 5

BOUNDARY_EXCLUSION_RADIUS = 0.05

LINE_SAMPLING_INTERVAL = 0.1

NUM_DIVERSE_PATHS = 64

DIVERSITY_SEARCH_RADIUS = 2.0
# ==========================================================================================


def get_all_scene_paths(data_dir):
    split_dir = os.path.join(data_dir, 'splits')
    all_scene_paths = []
    for i in range(1, 7):
        split_file = os.path.join(split_dir, f'area{i}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                paths = [line.strip() for line in f.readlines()]
                all_scene_paths.extend(paths)
        else:
            print(f"Warning: Split file not found at {split_file}")
    return all_scene_paths


def is_path_coplanar(p_start, p_end, kd_tree_scene, check_interval, radius, min_points):
    path_vector = p_end - p_start
    path_length = np.linalg.norm(path_vector)
    if path_length < check_interval: 
        return True

    num_check_points = int(path_length / check_interval)
    if num_check_points <= 1:
        return True

    check_points = np.linspace(p_start, p_end, num_check_points)[1:-1]
    if len(check_points) == 0:
        return True

    neighbors_counts = kd_tree_scene.query_ball_point(check_points, r=radius, return_length=True)

    if np.any(neighbors_counts < min_points):
        return False

    return True

def select_diverse_candidates(p_gt, candidate_coords, num_to_select):
    if len(candidate_coords) <= num_to_select:
        return np.arange(len(candidate_coords))

    pool_size = min(len(candidate_coords), num_to_select * 5)
    random_pool_indices = np.random.choice(len(candidate_coords), size=pool_size, replace=False)
    pool_coords = candidate_coords[random_pool_indices]
    
    direction_vectors = pool_coords - p_gt
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 
    unit_vectors = direction_vectors / norms

    selected_pool_indices = [0] 
    
    while len(selected_pool_indices) < num_to_select and len(selected_pool_indices) < len(pool_coords):
        best_idx = -1
        max_min_angle = -1
        
        selected_vectors = unit_vectors[selected_pool_indices]
        
        for idx in range(len(pool_coords)):
            if idx in selected_pool_indices:
                continue

            cos_sims = np.dot(selected_vectors, unit_vectors[idx])
            min_cos_sim = np.max(cos_sims) 
            angle_score = 1 - min_cos_sim  
            
            if angle_score > max_min_angle:
                max_min_angle = angle_score
                best_idx = idx
        
        if best_idx != -1:
            selected_pool_indices.append(best_idx)
        else:
            break

    final_indices = random_pool_indices[selected_pool_indices]
    
    return final_indices


def get_points_along_path(p_start, p_end, coords, kd_tree_scene, max_distance):
    segment_center = (p_start + p_end) / 2
    segment_half_length = np.linalg.norm(p_end - p_start) / 2
    search_radius = segment_half_length + max_distance

    candidate_indices = kd_tree_scene.query_ball_point(segment_center, r=search_radius)
    
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)

    candidate_coords = coords[candidate_indices]

    line_vec = p_end - p_start
    line_length = np.linalg.norm(line_vec)
    
    if line_length < 1e-8: 
        return np.array([], dtype=int)
    
    line_unit = line_vec / line_length

    start_to_candidates = candidate_coords - p_start

    projections = np.dot(start_to_candidates, line_unit)
    projections = np.clip(projections, 0, line_length)

    closest_points = p_start + projections[:, np.newaxis] * line_unit

    distances = np.linalg.norm(candidate_coords - closest_points, axis=1)

    valid_mask = distances <= max_distance
    valid_indices = np.array(candidate_indices)[valid_mask]
    
    return valid_indices


def propagate_labels(gt_coords, gt_labels, coords, kd_tree_scene, kd_tree_boundary, boundary_coords, args, pbar_desc="GT Points"):
    points_to_add_with_label = {}

    if len(gt_coords) == 0:
        return points_to_add_with_label

    gt_pbar = tqdm(range(len(gt_coords)), desc=pbar_desc, leave=False)
    for i in gt_pbar:
        p_gt = gt_coords[i]
        label_gt = gt_labels[i]

        candidate_indices_in_subset = kd_tree_boundary.query_ball_point(p_gt, r=args.diversity_search_radius)
        
        if len(candidate_indices_in_subset) == 0:
            continue
        
        candidate_coords = boundary_coords[candidate_indices_in_subset]
        
        diverse_subset_indices = select_diverse_candidates(p_gt, candidate_coords, args.num_diverse_paths)
        
        final_candidate_indices_in_subset = np.array(candidate_indices_in_subset)[diverse_subset_indices]
        
        for bd_idx_in_subset in final_candidate_indices_in_subset:
            p_bd_nearest = boundary_coords[bd_idx_in_subset]
            dist = np.linalg.norm(p_gt - p_bd_nearest)

            if dist > args.max_gt_bd_distance:
                continue

            if is_path_coplanar(p_gt, p_bd_nearest, kd_tree_scene,
                               args.check_point_interval,
                               args.density_check_radius,
                               args.min_density_points):
                
                path_vector = p_bd_nearest - p_gt
                path_length = dist

                safe_path_length = path_length - args.boundary_exclusion_radius
                if safe_path_length <= args.line_sampling_interval:
                    continue

                path_unit_vector = path_vector / path_length
                p_new_end = p_gt + path_unit_vector * safe_path_length
                
                num_steps = int(safe_path_length / args.line_sampling_interval)
                if num_steps > 0:
                    line_points = np.linspace(p_gt, p_new_end, num_steps)
                    
                    _, nearest_point_indices = kd_tree_scene.query(line_points, k=1)
                    
                    for point_idx in nearest_point_indices:
                        points_to_add_with_label[point_idx] = label_gt

    return points_to_add_with_label


def find_boundary_neighborhood(coords, boundary_mask, radius, batch_size=100000):
    boundary_coords = coords[boundary_mask]
    if len(boundary_coords) == 0:
        return np.zeros(len(coords), dtype=bool)

    kd_tree_boundary = cKDTree(boundary_coords)
    
    neighborhood_mask = np.zeros(len(coords), dtype=bool)
    
    num_points = len(coords)
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch_coords = coords[start:end]

        within_radius = dists <= radius
        neighborhood_mask[start:end] = within_radius
    
    return neighborhood_mask


def calculate_boundary_neighborhood_stats(coords, boundary_mask, pseudo_points_dict, true_labels, radius=0.25):

    neighborhood_mask = find_boundary_neighborhood(coords, boundary_mask, radius)

    total_points_in_neighborhood = np.sum(neighborhood_mask)
    
    if total_points_in_neighborhood == 0:
        return {
            'total_points': 0,
            'pseudo_count': 0,
            'coverage': 0.0,
            'correct_count': 0,
            'accuracy': 0.0
        }

    neighborhood_indices = np.where(neighborhood_mask)[0]
    pseudo_indices = set(pseudo_points_dict.keys())

    neighborhood_with_pseudo = neighborhood_indices[np.isin(neighborhood_indices, list(pseudo_indices))]
    pseudo_count_in_neighborhood = len(neighborhood_with_pseudo)
    
    coverage = pseudo_count_in_neighborhood / total_points_in_neighborhood if total_points_in_neighborhood > 0 else 0.0
    
    correct_count = 0
    if pseudo_count_in_neighborhood > 0:
        for idx in neighborhood_with_pseudo:
            pseudo_label = pseudo_points_dict[idx]
            true_label = true_labels[idx]
            if pseudo_label == true_label:
                correct_count += 1
    
    accuracy = correct_count / pseudo_count_in_neighborhood if pseudo_count_in_neighborhood > 0 else 0.0
    
        return {
        'total_points': total_points_in_neighborhood,
        'pseudo_count': pseudo_count_in_neighborhood,
        'coverage': coverage,
        'correct_count': correct_count,
        'accuracy': accuracy
    }


def main(args):
    total_rounds = args.num_propagation_rounds + 1
    if args.num_propagation_rounds > 0:
        pass


    if not os.path.isdir(args.input_data_dir):
        print(f"Error: Input data directory not found: {args.input_data_dir}")
        return
    if not os.path.exists(args.sampling_file_path):
        print(f"Error: Sampling file not found: {args.sampling_file_path}")
        return

    all_scene_relative_paths = get_all_scene_paths(args.input_data_dir)
    gt_indices_all_scenes = torch.load(args.sampling_file_path)

    final_pseudo_labels = {}
    
    global_boundary_stats = {}
    
    global_overall_stats = {}
    
    scene_pbar = tqdm(all_scene_relative_paths, desc="Processing Scenes")
    for scene_rel_path in scene_pbar:
        scene_pbar.set_postfix_str(scene_rel_path)
        scene_full_path = os.path.join(args.input_data_dir, scene_rel_path)

        if not os.path.exists(scene_full_path):
            continue

        try:
            scene_data = torch.load(scene_full_path)
            if scene_data.shape[1] < 9:
                continue
            coords = scene_data[:, :3]
            labels = scene_data[:, 6].astype(np.int32)
            projbound = scene_data[:, 8].astype(np.int32)
        except Exception:
            continue

        gt_indices = gt_indices_all_scenes.get(scene_rel_path)
        if gt_indices is None or len(gt_indices) == 0:
            continue
        gt_coords = coords[gt_indices]
        gt_labels = labels[gt_indices]

        boundary_indices = np.where(projbound == 1)[0]
        if len(boundary_indices) == 0:
            continue
        boundary_coords = coords[boundary_indices]

        kd_tree_scene = cKDTree(coords)
        kd_tree_boundary = cKDTree(boundary_coords)
        
        boundary_mask = np.zeros(len(coords), dtype=bool)
        boundary_mask[boundary_indices] = True
        
        all_rounds_points = {}
        round_stats = []
        
        all_historical_seed_coords = gt_coords.copy()
        
        points_round0 = propagate_labels(
            gt_coords, gt_labels, coords, kd_tree_scene, kd_tree_boundary, boundary_coords, args,
            pbar_desc="Round 0"
        )
        all_rounds_points.update(points_round0)
        
        num_correct_round0 = 0
        if args.num_propagation_rounds > 0 and len(points_round0) > 0:
            pseudo_indices = np.array(list(points_round0.keys()))
            pseudo_labels = np.array(list(points_round0.values()))
            true_labels_subset = labels[pseudo_indices]
            num_correct_round0 = np.sum(pseudo_labels == true_labels_subset)
        
        boundary_stats_round0 = calculate_boundary_neighborhood_stats(
            coords, boundary_mask, all_rounds_points, labels, radius=0.25
        )
        
        overall_stats_round0 = {
            'total_pseudo': len(all_rounds_points),
            'correct_pseudo': num_correct_round0,
            'total_points': len(coords)
        }
        
        round_stats.append((
            'Round 0 (GT)', 
            len(points_round0), 
            num_correct_round0, 
            0,
            boundary_stats_round0,
            overall_stats_round0
        ))
        
        for round_idx in range(1, args.num_propagation_rounds + 1):
            num_seeds_to_select = int(len(gt_indices) * args.seeds_per_round_ratio)
            
            selected_seeds = select_exploration_seeds_from_scene(
                coords=coords,
                labels=labels,
                known_point_coords=all_historical_seed_coords,
                kd_tree_scene=kd_tree_scene,
                kd_tree_boundary=kd_tree_boundary,
                num_seeds=num_seeds_to_select,
                args=args
            )
            
            if not selected_seeds:
                if round_stats:
                    last_round_stats = round_stats[-1]
                    last_name, last_num_pts, last_num_correct, last_num_seeds, last_boundary_stats, last_overall_stats = last_round_stats

                    for remaining_round_idx in range(round_idx, args.num_propagation_rounds + 1):
                        round_stats.append((
                            f'Round {remaining_round_idx}', 
                            0,
                            0,
                            0,
                            last_boundary_stats,
                            last_overall_stats
                        ))
                break
            
            seed_indices = np.array([s[0] for s in selected_seeds])
            seed_labels = np.array([s[1] for s in selected_seeds])
            seed_coords = coords[seed_indices]
            
            all_historical_seed_coords = np.vstack([all_historical_seed_coords, seed_coords])
            
            points_this_round = propagate_labels(
                seed_coords, seed_labels, coords, kd_tree_scene, kd_tree_boundary, boundary_coords, args,
                pbar_desc=f"Round {round_idx}"
            )
            
            all_rounds_points.update(points_this_round)
            
            num_correct_this_round = 0
            if len(points_this_round) > 0:
                pseudo_indices_this = np.array(list(points_this_round.keys()))
                pseudo_labels_this = np.array(list(points_this_round.values()))
                true_labels_this = labels[pseudo_indices_this]
                num_correct_this_round = np.sum(pseudo_labels_this == true_labels_this)
            
            boundary_stats_this_round = calculate_boundary_neighborhood_stats(
                coords, boundary_mask, all_rounds_points, labels, radius=0.25
            )
            
            accumulated_pseudo_indices = list(all_rounds_points.keys())
            accumulated_pseudo_labels = np.array([all_rounds_points[idx] for idx in accumulated_pseudo_indices])
            accumulated_true_labels = labels[accumulated_pseudo_indices]
            num_correct_accumulated = np.sum(accumulated_pseudo_labels == accumulated_true_labels)
            
            overall_stats_this_round = {
                'total_pseudo': len(all_rounds_points),
                'correct_pseudo': num_correct_accumulated,
                'total_points': len(coords)
            }
            
            round_stats.append((
                f'Round {round_idx}', 
                len(points_this_round), 
                num_correct_this_round, 
                len(selected_seeds),
                boundary_stats_this_round,
                overall_stats_this_round
            ))

        if all_rounds_points:
            final_indices = np.array(list(all_rounds_points.keys()), dtype=np.int64)
            final_labels = np.array(list(all_rounds_points.values()), dtype=np.int64)
            
            final_pseudo_labels[scene_rel_path] = {
                "indices": final_indices,
                "pseudo_labels": final_labels
            }
            
            for round_idx, round_info in enumerate(round_stats):
                _, _, _, _, boundary_stats, overall_stats = round_info
                
               
                if round_idx not in global_boundary_stats:
                    global_boundary_stats[round_idx] = []
                global_boundary_stats[round_idx].append(boundary_stats)
                
               
                if round_idx not in global_overall_stats:
                    global_overall_stats[round_idx] = []
                global_overall_stats[round_idx].append(overall_stats)

    output_dir = os.path.dirname(args.output_pseudo_label_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    torch.save(final_pseudo_labels, args.output_pseudo_label_file)
      
    if global_overall_stats:
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        
        overall_summary = []
        
        for round_idx in range(args.num_propagation_rounds + 1):
            if round_idx in global_overall_stats:
                stats_list = global_overall_stats[round_idx]
                
                total_pseudo = sum(s['total_pseudo'] for s in stats_list)
                correct_pseudo = sum(s['correct_pseudo'] for s in stats_list)
                total_points = sum(s['total_points'] for s in stats_list)
                
                accuracy = correct_pseudo / total_pseudo if total_pseudo > 0 else 0.0
                coverage = total_pseudo / total_points if total_points > 0 else 0.0
            else:

                if round_idx > 0 and overall_summary:
                    last_summary = overall_summary[-1]
                    total_pseudo = last_summary['total_pseudo']
                    correct_pseudo = last_summary['correct_pseudo']
                    total_points = last_summary['total_points']
                    accuracy = last_summary['accuracy']
                    coverage = last_summary['coverage']
                else:
                    total_pseudo = correct_pseudo = total_points = 0
                    accuracy = coverage = 0.0
            
            overall_summary.append({
                'round_idx': round_idx,
                'total_pseudo': total_pseudo,
                'correct_pseudo': correct_pseudo,
                'total_points': total_points,
                'accuracy': accuracy,
                'coverage': coverage,
                'has_data': round_idx in global_overall_stats
            })

        print(f"\n{'Round':<12} {'Pseudo/Total':<25} {'Coverage':<15} {'Accuracy':<15} {'Cov Growth':<15} {'Pseudo Growth':<20}")
        print(f"{'-'*12} {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*20}")
        
        for i, summary in enumerate(overall_summary):
            round_name = f"Round {summary['round_idx']}" if summary['round_idx'] > 0 else "Round 0 (GT)"

            if not summary['has_data'] and i > 0:
                round_name += " *"  
            
            pseudo_total = f"{summary['total_pseudo']:,}/{summary['total_points']:,}"
            coverage_pct = summary['coverage'] * 100
            accuracy_pct = summary['accuracy'] * 100
   
            if i == 0:
                cov_growth = "-"
                pseudo_growth = "-"
            else:
                prev_summary = overall_summary[i-1]
                
                cov_diff = (summary['coverage'] - prev_summary['coverage']) * 100
                if abs(cov_diff) < 0.01:  
                    cov_growth = "0pp"
                else:
                    cov_growth = f"{cov_diff:+.2f}pp"
                
                pseudo_diff = summary['total_pseudo'] - prev_summary['total_pseudo']
                if pseudo_diff == 0:
                    pseudo_growth = "0"
                elif prev_summary['total_pseudo'] > 0:
                    pseudo_pct_growth = (pseudo_diff / prev_summary['total_pseudo']) * 100
                    pseudo_growth = f"{pseudo_diff:+,} ({pseudo_pct_growth:+.1f}%)"
                else:
                    pseudo_growth = f"{pseudo_diff:+,}"
            
            print(f"{round_name:<12} {pseudo_total:<25} {coverage_pct:>6.2f}%        {accuracy_pct:>6.2f}%        {cov_growth:<15} {pseudo_growth:<20}")
        
        print(f"{'-'*12} {'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*20}")

        if len(overall_summary) > 1:
            final = overall_summary[-1]
            initial = overall_summary[0]
            total_cov_growth = (final['coverage'] - initial['coverage']) * 100
            total_pseudo_growth = final['total_pseudo'] - initial['total_pseudo']
            total_pseudo_pct = (total_pseudo_growth / initial['total_pseudo'] * 100) if initial['total_pseudo'] > 0 else 0
       
        round_summary = []
        max_round_with_data = max(global_boundary_stats.keys()) if global_boundary_stats else 0
        
        for round_idx in range(args.num_propagation_rounds + 1):
            if round_idx in global_boundary_stats:
                stats_list = global_boundary_stats[round_idx]

                total_pts = sum(s['total_points'] for s in stats_list)
                pseudo_pts = sum(s['pseudo_count'] for s in stats_list)
                correct_pts = sum(s['correct_count'] for s in stats_list)
                
                avg_coverage = pseudo_pts / total_pts if total_pts > 0 else 0.0
                avg_accuracy = correct_pts / pseudo_pts if pseudo_pts > 0 else 0.0
            else:

                if round_idx > 0 and round_summary:
                    last_summary = round_summary[-1]
                    total_pts = last_summary['total_pts']
                    pseudo_pts = last_summary['pseudo_pts']
                    correct_pts = last_summary['correct_pts']
                    avg_coverage = last_summary['coverage']
                    avg_accuracy = last_summary['accuracy']
                else:
                    total_pts = pseudo_pts = correct_pts = 0
                    avg_coverage = avg_accuracy = 0.0
            
            round_summary.append({
                'round_idx': round_idx,
                'total_pts': total_pts,
                'pseudo_pts': pseudo_pts,
                'correct_pts': correct_pts,
                'coverage': avg_coverage,
                'accuracy': avg_accuracy,
                'has_data': round_idx in global_boundary_stats
            })

        print(f"\n{'Round':<12} {'Pseudo/Total':<22} {'Coverage':<15} {'Accuracy':<15} {'Cov Growth':<15} {'Pseudo Growth':<15}")
        print(f"{'-'*12} {'-'*22} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

        for i, summary in enumerate(round_summary):
            round_name = f"Round {summary['round_idx']}" if summary['round_idx'] > 0 else "Round 0 (GT)"
            
            if not summary['has_data'] and i > 0:
                round_name += " *"  
            
            pseudo_total = f"{summary['pseudo_pts']}/{summary['total_pts']}"
            coverage_pct = summary['coverage'] * 100
            accuracy_pct = summary['accuracy'] * 100

            if i == 0:
                cov_growth = "-"
                pseudo_growth = "-"
            else:
                prev_summary = round_summary[i-1]
 
                cov_diff = (summary['coverage'] - prev_summary['coverage']) * 100
                if abs(cov_diff) < 0.01:  
                    cov_growth = "0pp"
                else:
                    cov_growth = f"{cov_diff:+.1f}pp"

                pseudo_diff = summary['pseudo_pts'] - prev_summary['pseudo_pts']
                if pseudo_diff == 0:
                    pseudo_growth = "0"
                elif prev_summary['pseudo_pts'] > 0:
                    pseudo_pct_growth = (pseudo_diff / prev_summary['pseudo_pts']) * 100
                    pseudo_growth = f"{pseudo_diff:+,} ({pseudo_pct_growth:+.1f}%)"
                else:
                    pseudo_growth = f"{pseudo_diff:+,}"
            
            print(f"{round_name:<12} {pseudo_total:<22} {coverage_pct:>6.1f}%        {accuracy_pct:>6.1f}%        {cov_growth:<15} {pseudo_growth:<15}")
        
        print(f"{'-'*12} {'-'*22} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

        if len(round_summary) > 1:
            final = round_summary[-1]
            initial = round_summary[0]
            total_cov_growth = (final['coverage'] - initial['coverage']) * 100
            total_pseudo_growth = final['pseudo_pts'] - initial['pseudo_pts']
            total_pseudo_pct = (total_pseudo_growth / initial['pseudo_pts'] * 100) if initial['pseudo_pts'] > 0 else 0
            

    if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default=INPUT_DATA_DIR)
    parser.add_argument('--sampling_file_path', type=str, default=SAMPLING_FILE_PATH)
    parser.add_argument('--output_pseudo_label_file', type=str, default=OUTPUT_PSEUDO_LABEL_FILE)
    parser.add_argument('--max_gt_bd_distance', type=float, default=MAX_GT_BD_DISTANCE)
    parser.add_argument('--check_point_interval', type=float, default=CHECK_POINT_INTERVAL)
    parser.add_argument('--density_check_radius', type=float, default=DENSITY_CHECK_RADIUS)
    parser.add_argument('--min_density_points', type=int, default=MIN_DENSITY_POINTS)
    parser.add_argument('--boundary_exclusion_radius', type=float, default=BOUNDARY_EXCLUSION_RADIUS)
    parser.add_argument('--line_sampling_interval', type=float, default=LINE_SAMPLING_INTERVAL)
    parser.add_argument('--num_diverse_paths', type=int, default=NUM_DIVERSE_PATHS)
    parser.add_argument('--diversity_search_radius', type=float, default=DIVERSITY_SEARCH_RADIUS)
    parser.add_argument('--seeds_per_round_ratio', type=float, default=SEEDS_PER_ROUND_RATIO)
    parser.add_argument('--seed_spacing_radius', type=float, default=SEED_SPACING_RADIUS)
    parser.add_argument('--seed_min_distance_from_boundary', type=float, default=SEED_MIN_DISTANCE_FROM_BOUNDARY)
    parser.add_argument('--seed_min_density_points', type=int, default=SEED_MIN_DENSITY_POINTS)
    parser.add_argument('--seed_density_radius', type=float, default=SEED_DENSITY_RADIUS)
    
    args = parser.parse_args()
    main(args) 