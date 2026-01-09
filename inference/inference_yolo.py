"""
YOLO v8 Inference + Solafune Submission Generation.

Usage:
    python inference_yolo.py
    python inference_yolo.py --model runs/segment/tree_canopy/weights/best.pt
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


# ==================== CONFIG ====================
class Config:
    DATA_DIR = Path("dataset")
    TEST_IMAGES_DIR = DATA_DIR / "evaluation_images"
    
    MODEL_PATH = Path("runs/segment/tree_canopy/weights/best.pt")
    SUBMISSION_PATH = Path("submissions/submission_yolo.json")
    
    CONFIDENCE = 0.25  # Detection confidence threshold
    IOU_THRESHOLD = 0.5  # NMS IoU threshold
    
    # Class mapping (YOLO class id -> Solafune class name)
    CLASS_MAP = {
        0: 'individual_tree',
        1: 'group_of_trees'
    }


config = Config()


def mask_to_polygon(mask, simplify=True):
    """
    Convert binary mask to polygon coords (flat list).
    
    Args:
        mask: Binary numpy array (H, W)
        simplify: Whether to simplify the polygon
        
    Returns:
        Flat list [x1, y1, x2, y2, ...] or None if invalid
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
    
    if simplify:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(contour) < 3:
        return None
    
    # Convert to flat coords - USE INTEGERS like sample_answer.json!
    flat_coords = []
    for pt in contour:
        flat_coords.extend([int(pt[0][0]), int(pt[0][1])])
    
    return flat_coords if len(flat_coords) >= 6 else None


def run_inference(model_path=None):
    print("\n" + "=" * 60)
    print("YOLO v8 INFERENCE - TREE CANOPY DETECTION")
    print("=" * 60)
    
    # Load model
    model_path = model_path or config.MODEL_PATH
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train YOLO first with: python train_yolo.py")
        return
    
    print(f"Model: {model_path}")
    print(f"Confidence: {config.CONFIDENCE}")
    print(f"IoU threshold: {config.IOU_THRESHOLD}")
    print("=" * 60 + "\n")
    
    model = YOLO(model_path)
    
    # Get test images
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif')) + \
                  list(config.TEST_IMAGES_DIR.glob('*.tiff'))
    
    print(f"Processing {len(test_images)} images...\n")
    
    submission = {'images': []}
    total_polygons = 0
    
    for img_path in tqdm(test_images, desc='Inference'):
        # Run inference
        results = model.predict(
            str(img_path),
            conf=config.CONFIDENCE,
            iou=config.IOU_THRESHOLD,
            verbose=False,
            retina_masks=True  # Higher quality masks
        )
        
        if not results or not results[0].masks:
            annotations = []
        else:
            result = results[0]
            annotations = []
            
            # Get image dimensions
            orig_h, orig_w = result.orig_shape
            
            # Process each detection
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Get class
                class_id = int(box.cls.item())
                class_name = config.CLASS_MAP.get(class_id, 'individual_tree')
                confidence = float(box.conf.item())
                
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Convert mask to polygon
                polygon = mask_to_polygon(mask_resized)
                
                if polygon:
                    annotations.append({
                        'class': class_name,
                        'confidence_score': round(confidence, 2),
                        'segmentation': polygon
                    })
        
        total_polygons += len(annotations)
        
        # Parse filename
        filename = img_path.name
        
        # Read actual image dimensions
        img = cv2.imread(str(img_path))
        if img is not None:
            orig_h, orig_w = img.shape[:2]
        else:
            orig_h, orig_w = 1024, 1024
        
        # Parse cm resolution from filename
        parts = filename.replace('.tif', '').replace('.tiff', '').split('_')
        cm_resolution = 40
        if parts[0].endswith('cm'):
            try:
                cm_resolution = int(parts[0].replace('cm', ''))
            except:
                pass
        
        scene_type = "agriculture_plantation" if cm_resolution == 10 else "mixed"
        
        submission['images'].append({
            'file_name': filename,
            'width': orig_w,
            'height': orig_h,
            'cm_resolution': cm_resolution,
            'scene_type': scene_type,
            'annotations': annotations
        })
    
    # Save submission
    config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f, indent=4)
    
    file_size = config.SUBMISSION_PATH.stat().st_size / (1024 * 1024)
    avg_per_image = total_polygons / len(submission['images']) if submission['images'] else 0
    
    print("\n" + "=" * 60)
    print("SUBMISSION CREATED!")
    print("=" * 60)
    print(f"File: {config.SUBMISSION_PATH}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Images: {len(submission['images'])}")
    print(f"Total polygons: {total_polygons}")
    print(f"Avg per image: {avg_per_image:.1f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='YOLO inference for tree canopy')
    parser.add_argument('--model', type=str, help='Path to model weights')
    args = parser.parse_args()
    
    run_inference(args.model)


if __name__ == '__main__':
    main()
