"""
YOLO v8 INFERENCE FOR GOOGLE COLAB - OPTIMIZED SUBMISSION
=========================================================

INSTRUCTIONS:
1. Train using colab_train_yolo.py first (or use your best.pt)
2. Upload evaluation_images to Drive
3. Run this to generate submission

REQUIRES:
- best.pt model (from training)
- evaluation_images folder in Drive
"""

# ============================================================
# CELL 1: SETUP
# ============================================================
# !pip install ultralytics opencv-python-headless tqdm
# from google.colab import drive, files
# drive.mount('/content/drive')

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CELL 2: CONFIGURATION
# ============================================================
class Config:
    # === PATHS (ADJUST FOR YOUR DRIVE) ===
    DRIVE_PATH = "/content/drive/MyDrive/solafune_tree"
    
    # Model path (after training)
    MODEL_PATH = "/content/runs/segment/tree_canopy_optimized/weights/best.pt"
    # Or from Drive: "/content/drive/MyDrive/best.pt"
    
    TEST_IMAGES_DIR = Path(DRIVE_PATH) / "evaluation_images"
    SUBMISSION_PATH = Path("/content/submission_yolo_optimized.json")
    
    # === INFERENCE SETTINGS ===
    CONFIDENCE = 0.15    # Lower = more detections (important for trees!)
    IOU_THRESHOLD = 0.5
    IMAGE_SIZE = 800     # Match training size
    
    # === TTA (Test-Time Augmentation) for better scores ===
    USE_TTA = True       # Flip augmentation during inference
    
    # === CLASS MAPPING ===
    CLASS_MAP = {
        0: 'individual_tree',
        1: 'group_of_trees'
    }


config = Config()


# ============================================================
# CELL 3: HELPER FUNCTIONS
# ============================================================
def mask_to_polygon(mask, simplify=True):
    """Convert binary mask to polygon coords (integers)."""
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
        # Simplify polygon
        epsilon = 0.003 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(contour) < 3:
        return None
    
    # INTEGER coordinates (required by Solafune!)
    flat_coords = []
    for pt in contour:
        flat_coords.extend([int(pt[0][0]), int(pt[0][1])])
    
    return flat_coords if len(flat_coords) >= 6 else None


def get_scene_type(filename):
    """Determine scene type based on filename."""
    if '10cm' in filename:
        return 'agriculture_plantation'
    elif '20cm' in filename:
        return 'residential_area'
    elif '40cm' in filename:
        return 'mixed'
    return 'mixed'


# ============================================================
# CELL 4: INFERENCE FUNCTION
# ============================================================
def run_inference():
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("YOLO v8 INFERENCE - OPTIMIZED")
    print("=" * 60)
    print(f"Model: {config.MODEL_PATH}")
    print(f"Confidence: {config.CONFIDENCE}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"TTA: {config.USE_TTA}")
    print("=" * 60 + "\n")
    
    if not Path(config.MODEL_PATH).exists():
        print(f"ERROR: Model not found at {config.MODEL_PATH}")
        return
    
    model = YOLO(config.MODEL_PATH)
    
    # Get test images
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif')) + \
                  list(config.TEST_IMAGES_DIR.glob('*.tiff'))
    
    print(f"Processing {len(test_images)} images...\n")
    
    submission = {'images': []}
    total_polygons = 0
    
    for img_path in tqdm(test_images, desc='Inference'):
        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        
        # Run inference
        predict_args = {
            'source': str(img_path),
            'conf': config.CONFIDENCE,
            'iou': config.IOU_THRESHOLD,
            'imgsz': config.IMAGE_SIZE,
            'verbose': False,
            'retina_masks': True,
        }
        
        if config.USE_TTA:
            # Test-Time Augmentation: horizontal flip
            predict_args['augment'] = True
        
        results = model.predict(**predict_args)
        
        annotations = []
        
        if results and results[0].masks is not None:
            result = results[0]
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for mask, box in zip(masks, boxes):
                class_id = int(box.cls.item())
                class_name = config.CLASS_MAP.get(class_id, 'individual_tree')
                confidence = float(box.conf.item())
                
                # Resize mask to original size
                mask_resized = cv2.resize(
                    mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
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
    
    # Save submission
    with open(config.SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f, indent=2)
    
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
    print("=" * 60)
    
    # Copy to Drive and download
    print("\nCopy to Drive:")
    print(f"!cp {config.SUBMISSION_PATH} /content/drive/MyDrive/")
    print("\nDownload directly:")
    print("from google.colab import files")
    print(f"files.download('{config.SUBMISSION_PATH}')")


# ============================================================
# CELL 5: RUN
# ============================================================
if __name__ == "__main__":
    run_inference()
