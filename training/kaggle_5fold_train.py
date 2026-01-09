"""
5-FOLD CROSS VALIDATION TRAINING - MAXIMUM DATA EFFICIENCY
===========================================================
Uses ALL 150 images for training (no data wasted!)
Each fold: 120 train, 30 val
Result: 5 models that together have seen ALL data

KAGGLE SETUP:
1. Add your dataset: solofune-tree
2. Enable GPU T4 x2
3. Run this script (~10 hours total, or 2hr per fold)
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm


class Config:
    # Kaggle paths
    DATASET_NAME = "solofune-tree"
    INPUT_PATH = Path(f"/kaggle/input/{DATASET_NAME}")
    SOURCE_IMAGES = INPUT_PATH / "train_images" / "train_images"  # Nested!
    SOURCE_ANNOTATIONS = INPUT_PATH / "train_annotations_updated.json"
    
    # Output
    OUTPUT_BASE = Path("/kaggle/working")
    
    # K-Fold settings
    N_FOLDS = 5
    SEED = 42
    
    # Training settings - optimized for T4 GPU
    MODEL = "yolov8s-seg"
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    EPOCHS = 200
    PATIENCE = 30
    
    CLASS_MAP = {'individual_tree': 0, 'group_of_trees': 1}


config = Config()


def convert_polygon_to_yolo(polygon, img_w, img_h):
    coords = []
    for i in range(0, len(polygon), 2):
        coords.extend([
            max(0.0, min(1.0, polygon[i] / img_w)),
            max(0.0, min(1.0, polygon[i + 1] / img_h))
        ])
    return coords


def create_fold_dataset(fold_idx, train_indices, val_indices, all_images):
    """Create YOLO dataset for one fold."""
    fold_dir = config.OUTPUT_BASE / f"fold_{fold_idx}"
    
    train_img = fold_dir / "images" / "train"
    val_img = fold_dir / "images" / "val"
    train_lbl = fold_dir / "labels" / "train"
    val_lbl = fold_dir / "labels" / "val"
    
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process train
    for idx in train_indices:
        img_data = all_images[idx]
        process_image(img_data, train_img, train_lbl)
    
    # Process val
    for idx in val_indices:
        img_data = all_images[idx]
        process_image(img_data, val_img, val_lbl)
    
    # Create YAML
    yaml_path = fold_dir / "data.yaml"
    yaml_content = f"""path: {fold_dir}
train: images/train
val: images/val
names:
  0: individual_tree
  1: group_of_trees
nc: 2
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def process_image(img_data, images_dir, labels_dir):
    filename = img_data['file_name']
    src_path = config.SOURCE_IMAGES / filename
    
    if not src_path.exists():
        return
    
    # Symlink instead of copy (faster)
    dst_path = images_dir / filename
    if not dst_path.exists():
        try:
            os.symlink(str(src_path), str(dst_path))
        except:
            shutil.copy(str(src_path), str(dst_path))
    
    # Create label
    img_w = img_data.get('width', 1024)
    img_h = img_data.get('height', 1024)
    label_path = labels_dir / filename.replace('.tif', '.txt').replace('.tiff', '.txt')
    
    lines = []
    for ann in img_data.get('annotations', []):
        class_id = config.CLASS_MAP.get(ann.get('class', 'individual_tree'), 0)
        polygon = ann.get('segmentation', [])
        if len(polygon) >= 6:
            coords = convert_polygon_to_yolo(polygon, img_w, img_h)
            lines.append(f"{class_id} " + ' '.join(f"{c:.6f}" for c in coords))
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))


def train_fold(fold_idx, yaml_path):
    """Train one fold."""
    from ultralytics import YOLO
    import torch
    
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    
    model = YOLO(config.MODEL)
    
    results = model.train(
        data=str(yaml_path),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        device=0,
        
        # Strong augmentation
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.4,
        degrees=20,
        translate=0.2,
        scale=0.6,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Training settings
        optimizer='AdamW',
        lr0=0.001,
        warmup_epochs=5,
        
        # Output
        project=str(config.OUTPUT_BASE / "runs"),
        name=f"fold_{fold_idx}",
        exist_ok=True,
        
        # Performance
        amp=True,
        workers=4,
    )
    
    # Save best model path
    best_path = config.OUTPUT_BASE / "runs" / f"fold_{fold_idx}" / "weights" / "best.pt"
    return str(best_path)


def run_kfold_training():
    """Main K-Fold training pipeline."""
    print("="*60)
    print("5-FOLD CROSS VALIDATION TRAINING")
    print("="*60)
    
    # Load annotations
    print(f"Loading: {config.SOURCE_ANNOTATIONS}")
    with open(config.SOURCE_ANNOTATIONS) as f:
        data = json.load(f)
    
    all_images = data['images']
    n_images = len(all_images)
    print(f"Total images: {n_images}")
    
    # Create folds
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    indices = np.arange(n_images)
    
    trained_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)} ---")
        
        # Create dataset for this fold
        yaml_path = create_fold_dataset(fold_idx, train_idx, val_idx, all_images)
        
        # Train
        model_path = train_fold(fold_idx, yaml_path)
        trained_models.append(model_path)
        
        print(f"Fold {fold_idx + 1} complete: {model_path}")
    
    # Save all model paths
    with open(config.OUTPUT_BASE / "kfold_models.json", 'w') as f:
        json.dump(trained_models, f, indent=2)
    
    print("\n" + "="*60)
    print("K-FOLD TRAINING COMPLETE!")
    print("="*60)
    print("Trained models:")
    for m in trained_models:
        print(f"  - {m}")
    print("\nNext: Run inference with all 5 models!")
    
    return trained_models


if __name__ == "__main__":
    run_kfold_training()
