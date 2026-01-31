import numpy as np

from uav_search.to_map_test import to_map_xyz

def obstacle_update(obstacle_map, start_position, depth_image, camera_fov, camera_position, camera_orientation):
    image_shape = (depth_image.shape[0], depth_image.shape[1])
    new_obstacle_map = obstacle_map.copy()
    
    map_size = np.array([200.0, 200.0, 50.0])
    grid_size = np.array([5.0, 5.0, 5.0])
    map_resolution = (map_size / grid_size).astype(int)

    start_pos_np = start_position.to_numpy_array()
    map_origin = start_pos_np - map_size / 2.0
    
    for v in range(image_shape[0]):
        for u in range(image_shape[1]):
            depth = depth_image[v, u]
            if depth >= 250:
                continue

            world_coords = to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation)
            map_coords = world_coords - map_origin
            grid_indices = (map_coords / grid_size).astype(int)
            gx, gy, gz = grid_indices

            if 0 <= gx < map_resolution[0] and 0 <= gy < map_resolution[1] and 0 <= gz < map_resolution[2]:
                new_obstacle_map[gx, gy, gz] = 1.0
    
    return new_obstacle_map

def _crop_rotate_and_pad(full_map: np.ndarray, center_coords: np.ndarray, crop_size: tuple, padding_value: float, orientation: int) -> np.ndarray:
    local_map = np.full(crop_size, padding_value, dtype=full_map.dtype)
    center_indices = np.round(center_coords).astype(int)

    map_dims = full_map.shape
    half_size = np.array(crop_size) // 2

    src_start = center_indices - half_size
    src_end = center_indices + half_size

    dest_start = np.zeros_like(half_size)
    dest_end = np.array(crop_size, dtype=int)

    for i in range(3):
        if src_start[i] < 0:
            dest_start[i] = -src_start[i]
            src_start[i] = 0

        if src_end[i] > map_dims[i]:
            dest_end[i] -= (src_end[i] - map_dims[i])
            src_end[i] = map_dims[i] 

    src_slice = tuple(slice(s, e) for s, e in zip(src_start, src_end))
    dest_slice = tuple(slice(s, e) for s, e in zip(dest_start, dest_end))

    if all(s.start < s.stop for s in src_slice):
        local_map[dest_slice] = full_map[src_slice]

    rotated_map = np.rot90(local_map, k=orientation, axes=(0, 1))
    return rotated_map

def map_input_preparation(attraction_map, exploration_map, obstacle_map, uav_pose: dict):
    position = np.array(uav_pose['position'])
    orientation = uav_pose['orientation']

    attraction_map_input = _crop_rotate_and_pad(
        full_map=attraction_map[:, :, :, 0],
        center_coords=position,
        crop_size=(20, 20, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    exploration_map_input = _crop_rotate_and_pad(
        full_map=exploration_map,
        center_coords=position,
        crop_size=(20, 20, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    obstacle_map_input = _crop_rotate_and_pad(
        full_map=obstacle_map,
        center_coords=position,
        crop_size=(8, 8, 4),
        padding_value=1.0,
        orientation=orientation
    )
    
    map_input = {
        'attraction_map_input': np.transpose(attraction_map_input,(2,0,1)),
        'exploration_map_input': np.transpose(exploration_map_input,(2,0,1)),
        'obstacle_map_input': np.transpose(obstacle_map_input,(2,0,1))
    }
    
    return map_input

def map_input_preparation_z(attraction_map, exploration_map, obstacle_map, uav_pose: dict):
    position = np.array(uav_pose['position'])
    orientation = uav_pose['orientation']

    attraction_map_input = _crop_rotate_and_pad(
        full_map=attraction_map[:, :, :, 0],
        center_coords=position,
        crop_size=(40, 40, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    exploration_map_input = _crop_rotate_and_pad(
        full_map=exploration_map,
        center_coords=position,
        crop_size=(40, 40, 10),
        padding_value=-1.0,
        orientation=orientation
    )

    obstacle_map_input = _crop_rotate_and_pad(
        full_map=obstacle_map,
        center_coords=position,
        crop_size=(40, 40, 10),
        padding_value=1.0,
        orientation=orientation
    )
    
    map_input = {
        'attraction_map_input': attraction_map_input,
        'exploration_map_input': exploration_map_input,
        'obstacle_map_input': obstacle_map_input
    }
    
    return map_input