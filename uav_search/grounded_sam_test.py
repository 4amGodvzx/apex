import time
import warnings
import torch
import numpy as np
from uav_search.api_test import generate_object_description

# Hyperparameters
box_threshold = 0.3
text_threshold = 0.3

def grounded_sam(qwen_processor, qwen_model, dino_processor, dino_model, sam_processor, sam_model, pil_image, rgb_base64, object_description):

    device = "cuda:1"
    warnings.filterwarnings("ignore")  
    
    time_start = time.time()
    
    result_dict = {
        "success": False,
        "masks": None,
        "scores": None,
    }
    
    mllm_result = generate_object_description(rgb_base64, object_description)
    if not mllm_result["success"]:
        return result_dict, []

    try:
        object_part, score_part = mllm_result["response"].split(";")
        text_labels = [[obj.strip() for obj in object_part.split(",")]]
        attraction_scores = [float(score.strip()) for score in score_part.split(",")]
    except Exception as e:
        return result_dict, []

    if len(text_labels[0]) != len(attraction_scores):
        return result_dict, []

    time_dino_start = time.time()
    
    inputs = dino_processor(images=pil_image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]]
    )

    time_dino_end = time.time()
    dino_result = results[0]

    all_masks = [[] for _ in range(len(text_labels[0]))]
    all_scores = [[] for _ in range(len(text_labels[0]))]

    label_to_index = {label: idx for idx, label in enumerate(text_labels[0])}

    time_sam_start = time.time()

    sam_inputs = sam_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeddings = sam_model.get_image_embeddings(sam_inputs["pixel_values"])

    input_points_list = []
    valid_indices_map = []

    original_w, original_h = pil_image.size
    scale = 1024.0 / max(original_w, original_h)

    for box, score, label in zip(dino_result["boxes"], dino_result["scores"], dino_result["labels"]):
        if label in label_to_index:
            if box[2] - box[0] < 20 and box[3] - box[1] < 20:
                continue
            
            idx = label_to_index[label]
            
            raw_center_x = (box[0] + box[2]) / 2
            raw_center_y = (box[1] + box[3]) / 2
            
            new_center_x = raw_center_x * scale
            new_center_y = raw_center_y * scale
            
            input_points_list.append([new_center_x.item(), new_center_y.item()])
            valid_indices_map.append(idx)
            
    if len(input_points_list) > 0:
        batch_size = len(input_points_list)
        
        input_points_tensor = torch.tensor(input_points_list, device=device).unsqueeze(1).unsqueeze(1)
        
        image_embeddings_expanded = image_embeddings.repeat(batch_size, 1, 1, 1)
        
        with torch.no_grad():
            sam_outputs = sam_model(
                image_embeddings=image_embeddings_expanded,
                input_points=input_points_tensor
            )
        
        original_sizes = sam_inputs["original_sizes"].repeat(batch_size, 1)
        reshaped_input_sizes = sam_inputs["reshaped_input_sizes"].repeat(batch_size, 1)
        
        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), 
            original_sizes.cpu(), 
            reshaped_input_sizes.cpu()
        )
        iou_scores = sam_outputs.iou_scores.cpu()
        
        for i, target_idx in enumerate(valid_indices_map):
            mask_tensor = masks[i][0] # (3, H, W)
            score_tensor = iou_scores[i][0] # (3,)
            
            best_idx = torch.argmax(score_tensor).item()
            final_mask = mask_tensor[best_idx].numpy()
            final_score = score_tensor[best_idx].item()
            
            all_masks[target_idx].append(final_mask)
            all_scores[target_idx].append(final_score)
            
    else:
        print("[Planning] SAM: No valid objects found to segment.")

    time_end = time.time()

    result_dict.update({
        "success": True,
        "masks": all_masks,
        "scores": all_scores
    })
    
    return result_dict, attraction_scores
