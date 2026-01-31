import airsim
import time
import warnings

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from uav_search.to_map_test import to_map_xyz

# Hyperparameters
detection_box_threshold = 0.3
detection_text_threshold = 0.7

def detection_test(processor, model, pil_image, depth_image, camera_info, camera_position, camera_orientation, object_name) :
    device = "cuda:1" 
    warnings.filterwarnings("ignore")

    processor = processor
    model = model

    text_label = [[object_name]]
    target_label = text_label[0][0].lower()

    time_start = time.time()

    inputs = processor(images=pil_image, text=text_label, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=detection_box_threshold,
        text_threshold=detection_text_threshold,
        target_sizes=[pil_image.size[::-1]]
    )
    
    time_end = time.time()    
    result = results[0]

    matched_indices = [i for i, label in enumerate(result["labels"]) if label.lower() == target_label]
    if not matched_indices:
        return None
    max_score_idx = matched_indices[result["scores"][matched_indices].argmax().item()]

    center_x = (result["boxes"][max_score_idx][0] + result["boxes"][max_score_idx][2]) / 2
    center_y = (result["boxes"][max_score_idx][1] + result["boxes"][max_score_idx][3]) / 2
    depth = depth_image[(center_y/2).int().item(), (center_x/2).int().item()]

    point_world = to_map_xyz(center_y.int().item(), center_x.int().item(), depth, pil_image.size[::-1], camera_info, camera_position, camera_orientation)
    print(f"[Detection] Detected object world coordinates: {point_world}")
    return point_world