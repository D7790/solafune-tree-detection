"""
5-FOLD ENSEMBLE INFERENCE WITH MULTI-SCALE TTA
===============================================
Uses all 5 fold models + multi-scale for maximum accuracy

Run after kaggle_5fold_train.py completes
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm


class Config:
    DATASET_NAME = "solofune-tree"
    INPUT_PATH = Path(f"/kaggle/input/{DATASET_NAME}")
    TEST_IMAGES_DIR = INPUT_PATH / "evaluation_images" / "evaluation_images"
    
    # 5 fold models (after training)
    MODEL_PATHS = [
        "/kaggle/working/runs/fold_0/weights/best.pt",
        "/kaggle/working/runs/fold_1/weights/best.pt",
        "/kaggle/working/runs/fold_2/weights/best.pt",
        "/kaggle/working/runs/fold_3/weights/best.pt",
        "/kaggle/working/runs/fold_4/weights/best.pt",
    ]
    
    # Multi-scale inference (TTA)
    SCALES = [512, 640, 768]
    
    CONFIDENCE = 0.20
    SUBMISSION_PATH = Path("/kaggle/working/submission_5fold_tta.json")
    
    CLASS_MAP = {0: 'individual_tree', 1: 'group_of_trees'}


config = Config()


def run_5fold_inference():
    from ultralytics import YOLO
    
    print("="*60)
    print("5-FOLD ENSEMBLE + MULTI-SCALE TTA INFERENCE")
    print("="*60)
    
    # Load all models
    models = []
    for path in config.MODEL_PATHS:
        if Path(path).exists():
            model = YOLO(path)
            model.to('cuda:0')
            models.append(model)
            print(f"Loaded: {path}")
        else:
            print(f"NOT FOUND: {path}")
    
    if not models:
        print("ERROR: No models found!")
        return
    
    print(f"\nUsing {len(models)} models × {len(config.SCALES)} scales = {len(models) * len(config.SCALES)} predictions per image")
    
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif'))
    print(f"Processing {len(test_images)} images\n")
    
    submission = {'images': []}
    
    for img_path in tqdm(test_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        seen_centers = set()
        annotations = []
        
        # Run ALL models at ALL scales
        for model in models:
            for scale in config.SCALES:
                results = model.predict(
                    str(img_path),
                    conf=config.CONFIDENCE,
                    imgsz=scale,
                    device=0,
                    verbose=False,
                    retina_masks=True
                )
                
                if results[0].masks is not None:
                    for mask, box in zip(results[0].masks.data.cpu().numpy(), results[0].boxes):
                        # Grid-based dedup
                        cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                        cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                        key = (cx // 15, cy // 15)
                        if key in seen_centers:
                            continue
                        seen_centers.add(key)
                        
                        # Convert mask to polygon
                        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(
                            mask_r.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            if len(c) >= 3:
                                coords = [coord for p in c for coord in (int(p[0][0]), int(p[0][1]))]
                                if len(coords) >= 6:
                                    annotations.append({
                                        'class': config.CLASS_MAP.get(int(box.cls.item()), 'individual_tree'),
                                        'confidence_score': round(float(box.conf.item()), 2),
                                        'segmentation': coords
                                    })
        
        # Build submission entry
        filename = img_path.name
        cm = int(filename.split('cm')[0]) if 'cm' in filename else 40
        scene = 'agriculture_plantation' if '10cm' in filename else 'residential_area' if '20cm' in filename else 'mixed'
        
        submission['images'].append({
            'file_name': filename,
            'width': w,
            'height': h,
            'cm_resolution': cm,
            'scene_type': scene,
            'annotations': annotations
        })
    
    # Save
    with open(config.SUBMISSION_PATH, 'w') as f:
        json.dump(submission, f)
    
    total = sum(len(img['annotations']) for img in submission['images'])
    print(f"\n{'='*60}")
    print("SUBMISSION CREATED!")
    print(f"{'='*60}")
    print(f"File: {config.SUBMISSION_PATH}")
    print(f"Total polygons: {total}")
    print(f"Avg per image: {total / len(submission['images']):.1f}")


if __name__ == "__main__":
    run_5fold_inference()
