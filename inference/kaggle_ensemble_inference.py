"""
ENSEMBLE INFERENCE FOR KAGGLE - COMBINE 3 MODELS
=================================================

Run this after training to generate submission.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURATION - KAGGLE PATHS
# ============================================================
class Config:
    # === KAGGLE PATHS ===
    DATASET_NAME = "solofune-tree"
    
    INPUT_PATH = Path(f"/kaggle/input/{DATASET_NAME}")
    TEST_IMAGES_DIR = INPUT_PATH / "evaluation_images" / "evaluation_images"
    
    # Uploaded model paths
    MODEL_PATHS = [
        "/kaggle/input/s1/pytorch/default/1/model_1_512_best.pt",
        "/kaggle/input/s1/pytorch/default/1/model_2_640_best.pt",
        "/kaggle/input/s1/pytorch/default/1/model_3_heavy_aug_best.pt",
    ]
    
    SUBMISSION_PATH = Path("/kaggle/working/submission_ensemble.json")
    
    IMAGE_SIZES = [512, 640, 512]
    CONFIDENCE = 0.10
    IOU_THRESHOLD = 0.5
    IOU_FUSION = 0.5
    
    CLASS_MAP = {0: 'individual_tree', 1: 'group_of_trees'}


config = Config()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def mask_to_polygon(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    
    epsilon = 0.003 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(contour) < 3:
        return None
    
    flat_coords = []
    for pt in contour:
        flat_coords.extend([int(pt[0][0]), int(pt[0][1])])
    
    return flat_coords if len(flat_coords) >= 6 else None


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def nms_polygons(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        remaining = []
        for det in detections:
            iou = mask_iou(best[3], det[3])
            if iou < iou_threshold:
                remaining.append(det)
        detections = remaining
    
    return keep


def get_scene_type(filename):
    if '10cm' in filename:
        return 'agriculture_plantation'
    elif '20cm' in filename:
        return 'residential_area'
    return 'mixed'


# ============================================================
# ENSEMBLE INFERENCE
# ============================================================
def run_ensemble_inference():
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("ENSEMBLE INFERENCE - KAGGLE")
    print("=" * 60)
    
    models = []
    for path in config.MODEL_PATHS:
        if Path(path).exists():
            model = YOLO(path)
            model.to('cuda:0')  # Move to GPU!
            models.append(model)
            print(f"Loaded to GPU: {path}")
        else:
            print(f"WARNING: Not found: {path}")
    
    if len(models) == 0:
        print("ERROR: No models found!")
        return
    
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif')) + \
                  list(config.TEST_IMAGES_DIR.glob('*.tiff'))
    
    submission = {'images': []}
    total_polygons = 0
    
    for img_path in tqdm(test_images, desc='Inference'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        
        all_detections = []
        
        for i, model in enumerate(models):
            imgsz = config.IMAGE_SIZES[i] if i < len(config.IMAGE_SIZES) else 512
            
            results = model.predict(
                str(img_path),
                conf=config.CONFIDENCE,
                iou=config.IOU_THRESHOLD,
                imgsz=imgsz,
                verbose=False,
                retina_masks=True,
                device=0  # Force GPU
            )
            
            if results and results[0].masks is not None:
                result = results[0]
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for mask, box in zip(masks, boxes):
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    mask_resized = cv2.resize(
                        mask, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    polygon = mask_to_polygon(mask_resized)
                    if polygon:
                        all_detections.append((
                            polygon, class_id, confidence, mask_resized > 0.5
                        ))
        
        fused = nms_polygons(all_detections, config.IOU_FUSION)
        
        annotations = []
        for polygon, class_id, confidence, _ in fused:
            annotations.append({
                'class': config.CLASS_MAP.get(class_id, 'individual_tree'),
                'confidence_score': round(confidence, 2),
                'segmentation': polygon
            })
        
        total_polygons += len(annotations)
        
        filename = img_path.name
        parts = filename.replace('.tif', '').replace('.tiff', '').split('_')
        cm_resolution = 40
        if parts[0].endswith('cm'):
            try:
                cm_resolution = int(parts[0].replace('cm', ''))
            except:
                pass
        
        submission['images'].append({
            'file_name': filename,
            'width': orig_w,
            'height': orig_h,
            'cm_resolution': cm_resolution,
            'scene_type': get_scene_type(filename),
            'annotations': annotations
        })
    
    with open(config.SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    file_size = config.SUBMISSION_PATH.stat().st_size / (1024 * 1024)
    avg_per_image = total_polygons / len(submission['images']) if submission['images'] else 0
    
    print("\n" + "=" * 60)
    print("SUBMISSION CREATED!")
    print("=" * 60)
    print(f"File: {config.SUBMISSION_PATH}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Total polygons: {total_polygons}")
    print(f"Avg per image: {avg_per_image:.1f}")


if __name__ == "__main__":
    run_ensemble_inference()
