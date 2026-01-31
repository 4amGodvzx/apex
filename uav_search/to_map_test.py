import airsim
import numpy as np
import time

from uav_search.airsim_utils import get_images

def quaternion_rotate_vector(quaternion, vector):
    q = np.array([quaternion.w_val, quaternion.x_val, quaternion.y_val, quaternion.z_val])
    v = np.array([vector.x_val, vector.y_val, vector.z_val])
    
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    rotated_v = np.dot(R, v)
    
    return airsim.Vector3r(rotated_v[0], rotated_v[1], rotated_v[2])

def to_map_xyz(v, u, depth, image_shape: tuple, camera_fov, camera_position, camera_orientation) -> np.ndarray:
    width = image_shape[1]
    height = image_shape[0]
    fov = camera_fov
    fx = fy = width / (2 * np.tan(np.deg2rad(fov / 2)))
    cx = width / 2
    cy = height / 2

    xn = (u - cx) / fx
    yn = (v - cy) / fy

    z_cam = depth
    x_cam = xn * z_cam
    y_cam = yn * z_cam


    point_camera = airsim.Vector3r(x_cam, y_cam, z_cam)
    point_ned = airsim.Vector3r(point_camera.z_val, point_camera.x_val, point_camera.y_val)

    rotated_point_world = quaternion_rotate_vector(camera_orientation, point_ned)
    point_world = rotated_point_world + camera_position

    return point_world.to_numpy_array()
