"""
LOCAL ENSEMBLE INFERENCE - Solafune IOU-based Polygon Format
=============================================================
Uses downloaded Kaggle models to generate submission.

REQUIREMENTS:
1. Put your 3 downloaded .pt files in models/ folder
2. Have evaluation_images in dataset/ folder
3. Run: python local_ensemble_inference.py
"""

import json
import cv2
import numpy as np
from pathlib import Path, PosixPath, WindowsPath
from tqdm import tqdm
import pathlib

# FIX: Allow loading Linux-trained models on Windows
pathlib.PosixPath = pathlib.WindowsPath


class Config:
    # All 3 models for ensemble (slower but better accuracy)
    MODEL_PATHS = [
        Path("models/model_1_512_best.pt"),
        Path("models/model_2_640_best.pt"),
        Path("models/model_3_heavy_aug_best.pt"),
    ]
    
    # Local paths
    TEST_IMAGES_DIR = Path("dataset/evaluation_images")
    SUBMISSION_PATH = Path("submissions/submission_ensemble_final.json")
    
    # Inference settings
    IMAGE_SIZES = [512, 640, 512]  # Match training sizes
    CONFIDENCE = 0.10              # Low = more detections
    IOU_THRESHOLD = 0.5
    IOU_FUSION = 0.5               # Merge overlapping detections
    
    # Solafune class mapping
    CLASS_MAP = {0: 'individual_tree', 1: 'group_of_trees'}


config = Config()


def mask_to_polygon(mask):
    """
    Convert binary mask to polygon with INTEGER coordinates.
    Solafune requires: flat list [x1, y1, x2, y2, ...] of integers.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    
    # Simplify polygon
    epsilon = 0.003 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(contour) < 3:
        return None
    
    # INTEGER coordinates - REQUIRED by Solafune!
    flat_coords = []
    for pt in contour:
        flat_coords.extend([int(pt[0][0]), int(pt[0][1])])
    
    return flat_coords if len(flat_coords) >= 6 else None


def mask_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def nms_polygons(detections, iou_threshold=0.5):
    """Non-Maximum Suppression for polygon detections."""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        # Remove overlapping detections
        detections = [d for d in detections if mask_iou(best[3], d[3]) < iou_threshold]
    
    return keep


def get_scene_type(filename):
    """Determine scene type from filename."""
    if '10cm' in filename:
        return 'agriculture_plantation'
    elif '20cm' in filename:
        return 'residential_area'
    elif '40cm' in filename:
        return 'mixed'
    return 'mixed'


def run_ensemble_inference():
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("LOCAL ENSEMBLE INFERENCE - SOLAFUNE FORMAT")
    print("=" * 60)
    
    # Load models TO GPU
    models = []
    for path in config.MODEL_PATHS:
        if path.exists():
            model = YOLO(str(path))
            model.to('cuda:0')  # Move to GPU!
            models.append(model)
            print(f"Loaded to GPU: {path}")
        else:
            print(f"NOT FOUND: {path}")
    
    if not models:
        print("\nERROR: No models found!")
        print("Put your downloaded .pt files in the 'models/' folder")
        return
    
    print(f"\nUsing {len(models)} models for ensemble")
    
    # Get test images
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif')) + \
                  list(config.TEST_IMAGES_DIR.glob('*.tiff'))
    
    if not test_images:
        print(f"\nERROR: No images found in {config.TEST_IMAGES_DIR}")
        return
    
    print(f"Found {len(test_images)} evaluation images")
    print("=" * 60 + "\n")
    
    submission = {'images': []}
    total_polygons = 0
    
    for img_path in tqdm(test_images, desc='Ensemble Inference'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        
        all_detections = []
        
        # Run each model
        for i, model in enumerate(models):
            imgsz = config.IMAGE_SIZES[i] if i < len(config.IMAGE_SIZES) else 512
            
            results = model.predict(
                str(img_path),
                conf=config.CONFIDENCE,
                iou=config.IOU_THRESHOLD,
                imgsz=imgsz,
                verbose=False,
                retina_masks=True,
                device=0  # Force GPU!
            )
            
            if results and results[0].masks is not None:
                result = results[0]
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for mask, box in zip(masks, boxes):
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    # Resize mask to original image size
                    mask_resized = cv2.resize(
                        mask, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    polygon = mask_to_polygon(mask_resized)
                    if polygon:
                        all_detections.append((
                            polygon, 
                            class_id, 
                            confidence, 
                            mask_resized > 0.5
                        ))
        
        # Apply NMS to fuse overlapping detections from all models
        fused = nms_polygons(all_detections, config.IOU_FUSION)
        
        # Build annotations in Solafune format
        annotations = []
        for polygon, class_id, confidence, _ in fused:
            annotations.append({
                'class': config.CLASS_MAP.get(class_id, 'individual_tree'),
                'confidence_score': round(confidence, 2),
                'segmentation': polygon  # Integer coordinates!
            })
        
        total_polygons += len(annotations)
        
        # Parse filename info
        filename = img_path.name
        parts = filename.replace('.tif', '').replace('.tiff', '').split('_')
        cm_resolution = 40
        if parts[0].endswith('cm'):
            try:
                cm_resolution = int(parts[0].replace('cm', ''))
            except:
                pass
        
        # Solafune format structure
        submission['images'].append({
            'file_name': filename,
            'width': orig_w,
            'height': orig_h,
            'cm_resolution': cm_resolution,
            'scene_type': get_scene_type(filename),
            'annotations': annotations
        })
    
    # Save submission
    config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
    file_size = config.SUBMISSION_PATH.stat().st_size / (1024 * 1024)
    avg = total_polygons / len(submission['images']) if submission['images'] else 0
    
    print("\n" + "=" * 60)
    print("SUBMISSION CREATED!")
    print("=" * 60)
    print(f"File: {config.SUBMISSION_PATH}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Images: {len(submission['images'])}")
    print(f"Total polygons: {total_polygons}")
    print(f"Avg per image: {avg:.1f}")
    print("=" * 60)
    print("\nReady to submit to Solafune!")


if __name__ == "__main__":
    run_ensemble_inference()
