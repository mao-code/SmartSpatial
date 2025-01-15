import os
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import clip
from collections import defaultdict
import scipy.stats

# Load YOLOv8 model
yolo = YOLO('yolov8x.pt')

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define IoU calculation function
def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(box1[0], box2[0])  # x1
    yA = max(box1[1], box2[1])  # y1
    xB = min(box1[2], box2[2])  # x2
    yB = min(box1[3], box2[3])  # y2

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_ap(precisions, recalls):
    """
    Compute the average precision (AP) using the 11-point interpolation method.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
        ap += p / 11.0
    return ap

def compute_ap_per_class(gt_boxes, detected_objects, iou_threshold=0.5):
    """
    Computes Average Precision (AP) per class for a single image.
    Returns a dictionary mapping class names to AP values and a list of IoUs of matched detections.
    """
    # Collect all classes
    classes = set([gt['caption'] for gt in gt_boxes] + [det['caption'] for det in detected_objects])
    ap_dict = {}
    iou_list = []
    for cls in classes:
        gt_cls_boxes = [gt['box'] for gt in gt_boxes if gt['caption'] == cls]
        det_cls_boxes = [det for det in detected_objects if det['caption'] == cls]

        # Sort detections by confidence
        det_cls_boxes = sorted(det_cls_boxes, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(det_cls_boxes))
        fp = np.zeros(len(det_cls_boxes))
        total_gt = len(gt_cls_boxes)
        gt_matched = [False] * total_gt

        for idx, det in enumerate(det_cls_boxes):
            box_det = det['box']
            max_iou = 0.0
            max_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_cls_boxes):
                iou = compute_iou(box_det, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            if max_iou >= iou_threshold:
                if not gt_matched[max_gt_idx]:
                    tp[idx] = 1
                    gt_matched[max_gt_idx] = True
                    iou_list.append(max_iou)
                else:
                    fp[idx] = 1  # Duplicate detection
            else:
                fp[idx] = 1  # False positive

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (total_gt + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        # Compute AP
        ap = compute_ap(precisions, recalls)
        ap_dict[cls] = ap

    return ap_dict, iou_list

def run_iou_clip(
    image_paths,
    prompt_datas,
    bbox_ref_mapping
):
    results_list = []
    global_idx = 0

    for idx, data in enumerate(prompt_datas):
        position = data['prompt_meta']['objects'][0]['pos']
        bbox_refs = bbox_ref_mapping[position]

        image_path = image_paths[global_idx]  # Ensure images are ordered correctly

        center_name = data['prompt_meta']['center']
        obj_name = data['prompt_meta']['objects'][0]['obj']

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Object detection using YOLOv8
        detections = yolo(image_path, verbose=False)
        detection = detections[0]
        boxes = detection.boxes.xyxy.cpu().numpy()  # Detected bounding boxes
        scores = detection.boxes.conf.cpu().numpy()  # Confidence scores
        labels = detection.boxes.cls.cpu().numpy().astype(int)  # Class labels

        # Map labels to class names
        class_names = yolo.names  # Mapping from class indices to names

        detected_objects = []
        for box, score, label in zip(boxes, scores, labels):
            class_name = class_names[label]
            detected_objects.append({
                'caption': class_name,
                'box': box,  # [x1, y1, x2, y2]
                'score': score
            })

        # Get ground truth boxes
        gt_boxes = []
        for ref in bbox_refs:
            caption = ref['caption']
            caption = center_name if caption == "box" else obj_name

            box = ref['box']
            x1 = box['x']
            y1 = box['y']
            x2 = x1 + box['w']
            y2 = y1 + box['h']

            gt_boxes.append({
                'caption': caption,
                'box': np.array([x1, y1, x2, y2])
            })

        # Compute per-class AP and collect IoUs
        ap_dict, iou_list = compute_ap_per_class(gt_boxes, detected_objects, iou_threshold=0.5)
        # Compute per-image mAP
        per_image_map = np.mean(list(ap_dict.values())) if ap_dict else 0.0
        # Compute per-image mean IoU
        per_image_mean_iou = np.mean(iou_list) if iou_list else 0.0

        # CLIP score computation
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([data['prompt']]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        clip_score = (image_features @ text_features.T).cpu().item()

        # Store results
        results_list.append({
            'image_path': image_path,
            'position': position,
            'per_image_map': per_image_map,
            'per_image_mean_iou': per_image_mean_iou,
            'clip_score': clip_score
        })

        global_idx += 1

    return results_list

def average_results(results_list):
    # Calculate average metrics
    total_iou = sum(res['per_image_mean_iou'] for res in results_list)
    total_map = sum(res['per_image_map'] for res in results_list)
    total_clip_score = sum(res['clip_score'] for res in results_list)
    num_samples = len(results_list)

    avg_iou = total_iou / num_samples
    avg_map = total_map / num_samples
    avg_clip_score = total_clip_score / num_samples

    # print(f'Result List: {results_list}')
    # print(f'Average per-image mean IoU: {avg_iou:.4f}')
    # print(f'Average per-image mAP@0.5: {avg_map:.4f}')
    # print(f'Average CLIP score: {avg_clip_score:.4f}')

    return avg_iou, avg_map, avg_clip_score

def compute_statistics(values, confidence=0.95):
    """Compute mean, std, and the confidence interval (two-sided) for a list of values."""
    arr = np.array(values)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)  # sample standard deviation

    n = len(arr)
    if n > 1:
        # t-distribution critical value
        alpha = 1 - confidence
        t_crit = scipy.stats.t.ppf(1 - alpha/2, n - 1)
        margin_of_error = t_crit * (std_val / np.sqrt(n))
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error
    else:
        # Not enough data to compute CI
        ci_lower, ci_upper = mean_val, mean_val

    return mean_val, std_val, ci_lower, ci_upper
