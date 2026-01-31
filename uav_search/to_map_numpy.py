import airsim
import numpy as np
import time

def depth_image_to_world_points(depth_image: np.ndarray, camera_fov: float, 
                                camera_position: airsim.Vector3r, 
                                camera_orientation: airsim.Quaternionr) -> np.ndarray:

    height, width = depth_image.shape
    fx = width / (2 * np.tan(np.deg2rad(camera_fov / 2)))
    fy = fx 
    cx = width / 2
    cy = height / 2

    u = np.arange(width)
    v = np.arange(height)
    u_map, v_map = np.meshgrid(u, v)

    xn = (u_map - cx) / fx
    yn = (v_map - cy) / fy

    z_cam = depth_image
    x_cam = xn * z_cam
    y_cam = yn * z_cam

    points_camera_ned = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_flat = points_camera_ned.reshape(-1, 3)

    q = camera_orientation.to_numpy_array()
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

    rotated_points_flat = points_flat @ R.T

    camera_pos_arr = camera_position.to_numpy_array()
    world_points = rotated_points_flat + camera_pos_arr

    return world_points