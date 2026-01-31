import airsim
import time
import numpy as np
from skimage.transform import resize
from collections import defaultdict, Counter

from uav_search.to_map_test import to_map_xyz
from uav_search.to_map_numpy import depth_image_to_world_points

def add_masks(masks):
    added_masks = []
    for mask_group in masks:
        if not mask_group:
            added_masks.append(None)
        else:
            stacked = np.stack(mask_group, axis=0)  # shape: [N, H, W]
            combined_mask = np.any(stacked, axis=0)
            added_masks.append(combined_mask)
    return added_masks

def downsample_masks(masks, scale_factor):
    downsampled = []
    for mask in masks:
        if mask is not None:
            downsampled_mask = resize(mask, (mask.shape[0] // scale_factor, mask.shape[1] // scale_factor), order=0, anti_aliasing=False)
            downsampled.append(downsampled_mask)
        else:
            downsampled.append(None)
    return downsampled

import numpy as np

def exploration_rate(distance: np.ndarray, max_depth=50, decay_factor=0.05, gain=10.0) -> np.ndarray:
    rates = np.where(
        distance > max_depth, 
        0.0, 
        np.exp(-decay_factor * (distance / max_depth)) * gain
    )
    return rates

def trace_rays_vectorized(drone_world_pos, endpoints_world, map_origin, grid_size, map_resolution):

    start_points = np.expand_dims(drone_world_pos, axis=0)
    
    directions = endpoints_world - start_points
    distances = np.linalg.norm(directions, axis=1)

    if np.all(distances == 0) or distances.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    max_distance = np.max(distances)
    step_size = grid_size[0] * 0.5
    num_steps = int(np.ceil(max_distance / step_size))
    
    t_values = np.linspace(0.0, 1.0, num_steps)
    
    sampled_points = start_points[:, np.newaxis, :] + \
                     directions[:, np.newaxis, :] * t_values[np.newaxis, :, np.newaxis]
    
    map_coords = sampled_points.reshape(-1, 3) - map_origin
    grid_indices = (map_coords / grid_size).astype(np.int32)
    
    valid_mask = np.all((grid_indices >= 0) & (grid_indices < np.array(map_resolution)), axis=1)
    grid_indices = grid_indices[valid_mask]
    
    Rx, Ry, Rz = map_resolution
    unique_ids = grid_indices[:, 0] * Ry * Rz + grid_indices[:, 1] * Rz + grid_indices[:, 2]
    _, unique_idx = np.unique(unique_ids, return_index=True)
    unique_grid_indices = grid_indices[unique_idx]
    
    return unique_grid_indices

def map_update(attraction_map, exploration_map, prepared_masks, attraction_scores, start_position, depth_image, camera_fov, camera_position, camera_orientation):
    depth_image = depth_image
    camera_fov = camera_fov
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    
    new_attraction_map = attraction_map.copy()
    new_exploration_map = exploration_map.copy()

    map_size = np.array([200.0, 200.0, 50.0])
    grid_size = np.array([5.0, 5.0, 5.0])
    map_resolution = (map_size / grid_size).astype(int)

    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0

    grid_contributions = defaultdict(list)

    for object_id, mask in enumerate(prepared_masks):
        if mask is None:
            continue

        pixels_v, pixels_u = np.where(mask)

        for v, u in zip(pixels_v, pixels_u):
            depth = depth_image[v, u]
            if depth >= 250:
                continue
            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)

            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                grid_contributions[(gx, gy, gz)].append((depth, object_id))

    for (gx, gy, gz), points_list in grid_contributions.items():
        if not points_list:
            continue

        min_depth_in_frame = min(p[0] for p in points_list)
        
        stored_depth = attraction_map[gx, gy, gz, 1]
        if min_depth_in_frame < stored_depth or stored_depth < 0:
            new_attraction_map[gx, gy, gz, 1] = min_depth_in_frame

        object_ids = [p[1] for p in points_list]
        if object_ids:
            winner_object_id = Counter(object_ids).most_common(1)[0][0]
            winning_score = attraction_scores[winner_object_id]
            new_attraction_map[gx, gy, gz, 0] = winning_score
    
    add_exploration_map = np.zeros_like(exploration_map)
    drone_world_pos = camera_position.to_numpy_array()

    valid_depth_mask = depth_image < 250
    
    all_endpoints_world = depth_image_to_world_points(
        depth_image, camera_fov, camera_position, camera_orientation
    )

    flat_mask = valid_depth_mask.flatten()
    valid_endpoints = all_endpoints_world[flat_mask]
    drone_world_pos = camera_position.to_numpy_array().flatten()

    traversed_grids_indices = trace_rays_vectorized(
        drone_world_pos, valid_endpoints, map_origin, grid_size, map_resolution
    )

    if traversed_grids_indices.shape[0] == 0:
        return new_attraction_map, new_exploration_map, 0
    
    grid_centers_world = (traversed_grids_indices + 0.5) * grid_size + map_origin
    distances_to_grids = np.linalg.norm(grid_centers_world - drone_world_pos, axis=1)

    rates = exploration_rate(distances_to_grids, max_depth=50, decay_factor=3, gain=10.0)

    gx, gy, gz = traversed_grids_indices.T
    current_rates = exploration_map[gx, gy, gz]
    update_mask = rates > current_rates
    add_exploration_map[gx[update_mask], gy[update_mask], gz[update_mask]] = rates[update_mask]
    
    new_exploration_map = exploration_map + add_exploration_map

    return new_attraction_map, new_exploration_map, 0

def map_update_simple(attraction_map, exploration_map, prepared_masks, attraction_scores, start_position, depth_image, camera_fov, camera_position, camera_orientation, ori):
    depth_image = depth_image
    camera_fov = camera_fov
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    
    new_attraction_map = attraction_map.copy()
    new_exploration_map = exploration_map.copy()

    map_size = np.array([200.0, 200.0, 50.0])
    grid_size = np.array([5.0, 5.0, 5.0])
    map_resolution = (map_size / grid_size).astype(int)

    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0

    grid_contributions = defaultdict(list)

    for object_id, mask in enumerate(prepared_masks):
        if mask is None:
            continue

        pixels_v, pixels_u = np.where(mask)

        for v, u in zip(pixels_v, pixels_u):
            depth = depth_image[v, u]
            if depth >= 250:
                continue

            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)

            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                grid_contributions[(gx, gy, gz)].append((depth, object_id))

    for (gx, gy, gz), points_list in grid_contributions.items():
        if not points_list:
            continue

        min_depth_in_frame = min(p[0] for p in points_list)
        
        stored_depth = attraction_map[gx, gy, gz, 1]
        if min_depth_in_frame < stored_depth or stored_depth < 0:
            new_attraction_map[gx, gy, gz, 1] = min_depth_in_frame

        object_ids = [p[1] for p in points_list]
        if object_ids:
            winner_object_id = Counter(object_ids).most_common(1)[0][0]
            winning_score = attraction_scores[winner_object_id]
            new_attraction_map[gx, gy, gz, 0] = winning_score
    
    VIEW_DEPTH = 30.0
    VIEW_HEIGHT = 20.0
    EXPLORATION_GAIN = 0.5
    EXPLORATION_ACCELERATION_FACTOR = 0.2

    drone_world_pos = camera_position.to_numpy_array().flatten()

    indices_x, indices_y, indices_z = np.indices(map_resolution)
    all_indices = np.stack([indices_x.ravel(), indices_y.ravel(), indices_z.ravel()], axis=-1)
    all_centers_world = (all_indices + 0.5) * grid_size + map_origin

    relative_coords = all_centers_world - drone_world_pos
    rel_x, rel_y, rel_z = relative_coords.T

    ori_conditions = [ori == 0, ori == 1, ori == 2, ori == 3]
    local_x_choices = [-rel_y, -rel_x, rel_y, rel_x]
    local_y_choices = [rel_x, -rel_y, -rel_x, rel_y]
    local_x = np.select(ori_conditions, local_x_choices)
    local_y = np.select(ori_conditions, local_y_choices)

    in_height_mask = np.abs(rel_z) <= VIEW_HEIGHT / 2
    in_depth_mask = (local_y > 0) & (local_y <= VIEW_DEPTH)
    in_angle_mask = np.abs(local_x) <= local_y
    in_prism_mask = in_height_mask & in_depth_mask & in_angle_mask

    traversed_indices_prism = all_indices[in_prism_mask]
    if traversed_indices_prism.shape[0] > 0:
        gx, gy, gz = traversed_indices_prism.T
        old_exploration_values = exploration_map[gx, gy, gz]
        distances_for_update = np.linalg.norm(relative_coords[in_prism_mask], axis=1)
        distances_for_update = np.clip(distances_for_update, 0, VIEW_DEPTH)

        exploration_increase = (1 - distances_for_update / VIEW_DEPTH) * EXPLORATION_GAIN

        new_exploration_map[gx, gy, gz] += exploration_increase
    # new_exploration_map *= 0.95

    return new_attraction_map, new_exploration_map, 0