import airsim
from PIL import Image
import numpy as np
import base64
import time

def get_images(client: airsim.MultirotorClient) -> tuple:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])

    rgb_response = responses[0]
    depth_response = responses[1]

    img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(rgb_response.height, rgb_response.width, 3)
    pil_image = Image.fromarray(img_rgb)
    
    depth_image = airsim.get_pfm_array(depth_response)

    camera_position = depth_response.camera_position
    camera_orientation = depth_response.camera_orientation

    rgb_vis = responses[2].image_data_uint8
    b64_image = base64.b64encode(rgb_vis).decode("utf-8")

    return pil_image, depth_image, camera_position, camera_orientation, rgb_vis, b64_image

def get_train_images(client: airsim.MultirotorClient) -> tuple:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
    ])

    depth_response = responses[0]
    
    depth_image = airsim.get_pfm_array(depth_response)

    camera_position = depth_response.camera_position
    camera_orientation = depth_response.camera_orientation

    return depth_image, camera_position, camera_orientation

def get_eval_images(client: airsim.MultirotorClient) -> tuple:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),
    ])
    
    rgb_vis = responses[0].image_data_uint8
    b64_image = base64.b64encode(rgb_vis).decode("utf-8")
    
    return b64_image