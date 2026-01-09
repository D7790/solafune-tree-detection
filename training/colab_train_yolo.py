"""
YOLO v8 TRAINING FOR GOOGLE COLAB - OPTIMIZED FOR 0.60+ IoU
============================================================

INSTRUCTIONS:
1. Upload your dataset folder to Google Drive
2. Mount Drive in Colab
3. Run this notebook cell by cell

Dataset structure needed:
your_drive/
└── solafune_tree/
    ├── train_images/          (150 .tif files)
    ├── train_annotations.json
    └── evaluation_images/     (150 .tif files)
"""

# ============================================================
# CELL 1: SETUP & INSTALL
# ============================================================
# !pip install ultralytics opencv-python-headless tqdm albumentations
# from google.colab import drive
# drive.mount('/content/drive')

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CELL 2: CONFIGURATION - ADJUST THESE!
# ============================================================
class Config:
    # === PATHS (ADJUST FOR YOUR DRIVE) ===
    DRIVE_PATH = "/content/drive/MyDrive/solafune_tree"
    SOURCE_IMAGES = Path(DRIVE_PATH) / "train_images"
    SOURCE_ANNOTATIONS = Path(DRIVE_PATH) / "train_annotations_updated.json"
    
    # Output will be created in Colab's local storage (faster)
    OUTPUT_DIR = Path("/content/yolo_dataset")
    
    # === TRAINING SETTINGS - MAXIMUM ACCURACY FOR 0.60 IoU ===
    
    # LARGE model for BEST accuracy - reduce batch/imgsz to fit in GPU
    MODEL = "yolov8l-seg"  # LARGE model = highest accuracy
    # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg
    
    IMAGE_SIZE = 800      # Balanced: good detail, fits in memory
    BATCH_SIZE = 2        # Small batch = LARGE model fits! (gradient accumulation helps)
    EPOCHS = 300          # More epochs for better convergence
    PATIENCE = 50         # More patience for large model
    
    # === DATA SPLIT ===
    TRAIN_RATIO = 0.85
    SEED = 42
    
    # === CLASS MAPPING ===
    CLASS_MAP = {
        'individual_tree': 0,
        'group_of_trees': 1
    }


config = Config()


# ============================================================
# CELL 3: DATA CONVERSION FUNCTION
# ============================================================
def convert_polygon_to_yolo(polygon_coords, img_width, img_height):
    """Convert flat polygon coords to YOLO normalized format."""
    normalized = []
    for i in range(0, len(polygon_coords), 2):
        x = polygon_coords[i] / img_width
        y = polygon_coords[i + 1] / img_height
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        normalized.extend([x, y])
    return normalized


def process_image(img_data, images_dir, labels_dir, source_images):
    """Process a single image and its annotations."""
    filename = img_data['file_name']
    src_path = source_images / filename
    
    if not src_path.exists():
        return False
    
    # Copy image
    dst_path = images_dir / filename
    shutil.copy(src_path, dst_path)
    
    # Get dimensions
    img_width = img_data.get('width', 1024)
    img_height = img_data.get('height', 1024)
    
    # Create label file
    label_name = filename.replace('.tif', '.txt').replace('.tiff', '.txt')
    label_path = labels_dir / label_name
    
    lines = []
    for ann in img_data.get('annotations', []):
        class_name = ann.get('class', 'individual_tree')
        class_id = config.CLASS_MAP.get(class_name, 0)
        
        polygon = ann.get('segmentation', [])
        if len(polygon) < 6:
            continue
        
        normalized = convert_polygon_to_yolo(polygon, img_width, img_height)
        coords_str = ' '.join([f"{c:.6f}" for c in normalized])
        lines.append(f"{class_id} {coords_str}")
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return True


def convert_to_yolo():
    """Convert Solafune dataset to YOLO format."""
    print("=" * 60)
    print("CONVERTING SOLAFUNE TO YOLO FORMAT")
    print("=" * 60)
    
    # Create directories
    train_images = config.OUTPUT_DIR / "images" / "train"
    val_images = config.OUTPUT_DIR / "images" / "val"
    train_labels = config.OUTPUT_DIR / "labels" / "train"
    val_labels = config.OUTPUT_DIR / "labels" / "val"
    
    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\nLoading: {config.SOURCE_ANNOTATIONS}")
    with open(config.SOURCE_ANNOTATIONS) as f:
        data = json.load(f)
    
    images = data['images']
    print(f"Found {len(images)} images")
    
    # Shuffle and split
    random.seed(config.SEED)
    random.shuffle(images)
    
    split_idx = int(len(images) * config.TRAIN_RATIO)
    train_data = images[:split_idx]
    val_data = images[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Process
    print("\nProcessing training data...")
    for img_data in tqdm(train_data):
        process_image(img_data, train_images, train_labels, config.SOURCE_IMAGES)
    
    print("Processing validation data...")
    for img_data in tqdm(val_data):
        process_image(img_data, val_images, val_labels, config.SOURCE_IMAGES)
    
    # Create YAML
    yaml_path = config.OUTPUT_DIR / "tree_canopy.yaml"
    yaml_content = f"""# Tree Canopy Detection - Optimized for Solafune
path: {config.OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: individual_tree
  1: group_of_trees

nc: 2
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"Dataset: {config.OUTPUT_DIR}")
    print(f"YAML: {yaml_path}")
    print(f"{'='*60}")
    
    return yaml_path


# ============================================================
# CELL 4: TRAINING FUNCTION
# ============================================================
def train_yolo(yaml_path):
    """Train YOLO with optimized settings for 0.60+ IoU."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print(f"TRAINING: {config.MODEL}")
    print("=" * 60)
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60 + "\n")
    
    model = YOLO(config.MODEL)
    
    results = model.train(
        data=str(yaml_path),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        device=0,  # GPU
        
        # ========== OPTIMIZATIONS FOR BETTER SCORES ==========
        
        # Multi-scale training (important for trees of different sizes!)
        scale=0.5,  # ±50% scale variation
        
        # Strong augmentations
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,  # Important for instance segmentation!
        flipud=0.5,
        fliplr=0.5,
        degrees=15,      # Rotation
        translate=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Training stability
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        # Regularization
        dropout=0.1,
        
        # Output
        project="/content/runs/segment",
        name="tree_canopy_optimized",
        exist_ok=True,
        
        # Performance
        amp=True,
        workers=4,
        
        # Saving
        save=True,
        save_period=20,
        plots=True,
        val=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Best model: /content/runs/segment/tree_canopy_optimized/weights/best.pt")
    print("\nCopy to Drive:")
    print("!cp /content/runs/segment/tree_canopy_optimized/weights/best.pt /content/drive/MyDrive/")
    print("=" * 60)
    
    return results


# ============================================================
# CELL 5: RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    # Step 1: Convert data
    yaml_path = convert_to_yolo()
    
    # Step 2: Train
    train_yolo(yaml_path)
