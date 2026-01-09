"""
UPDATED ENSEMBLE YOLO TRAINING - GPU & DRIVE ENABLED
==========================================================
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

class Config:
    DRIVE_PATH = "/content/drive/MyDrive/solafune_tree"
    SOURCE_IMAGES = Path(DRIVE_PATH) / "train_images"
    SOURCE_ANNOTATIONS = Path(DRIVE_PATH) / "train_annotations_updated.json"
    
    # DATA stays on local disk for speed during training
    OUTPUT_DIR = Path("/content/yolo_dataset")
    
    # SAVING results directly to Drive so lunch is safe!
    SAVE_DIR = "/content/drive/MyDrive/solafune_tree/runs/ensemble"
    
    MODELS = [
        {"name": "model_1_512", "arch": "yolov8s-seg", "imgsz": 512, "batch": 16, "epochs": 150, "augment": "default"},
        {"name": "model_2_640", "arch": "yolov8s-seg", "imgsz": 640, "batch": 12, "epochs": 150, "augment": "default"},
        {"name": "model_3_heavy_aug", "arch": "yolov8s-seg", "imgsz": 512, "batch": 16, "epochs": 150, "augment": "heavy"}
    ]
    
    TRAIN_RATIO = 0.85
    SEED = 42
    CLASS_MAP = {'individual_tree': 0, 'group_of_trees': 1}

config = Config()

# [Data conversion functions remain the same as your original script]
def convert_polygon_to_yolo(polygon_coords, img_width, img_height):
    normalized = []
    for i in range(0, len(polygon_coords), 2):
        x = max(0.0, min(1.0, polygon_coords[i] / img_width))
        y = max(0.0, min(1.0, polygon_coords[i + 1] / img_height))
        normalized.extend([x, y])
    return normalized

def convert_to_yolo():
    print("CONVERTING DATA...")
    train_images = config.OUTPUT_DIR / "images" / "train"
    val_images = config.OUTPUT_DIR / "images" / "val"
    train_labels = config.OUTPUT_DIR / "labels" / "train"
    val_labels = config.OUTPUT_DIR / "labels" / "val"
    for d in [train_images, val_images, train_labels, val_labels]: d.mkdir(parents=True, exist_ok=True)
    with open(config.SOURCE_ANNOTATIONS) as f: data = json.load(f)
    images = data['images']
    random.seed(config.SEED); random.shuffle(images)
    split_idx = int(len(images) * config.TRAIN_RATIO)
    train_data, val_data = images[:split_idx], images[split_idx:]
    for img_data in tqdm(train_data, desc="Train"): process_image(img_data, train_images, train_labels)
    for img_data in tqdm(val_data, desc="Val"): process_image(img_data, val_images, val_labels)
    yaml_path = config.OUTPUT_DIR / "tree_canopy.yaml"
    yaml_content = f"path: {config.OUTPUT_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: individual_tree\n  1: group_of_trees\nnc: 2"
    with open(yaml_path, 'w') as f: f.write(yaml_content)
    return yaml_path

def process_image(img_data, images_dir, labels_dir):
    filename = img_data['file_name']
    src_path = config.SOURCE_IMAGES / filename
    if not src_path.exists(): return
    shutil.copy(src_path, images_dir / filename)
    img_w, img_h = img_data.get('width', 1024), img_data.get('height', 1024)
    label_path = labels_dir / filename.replace('.tif', '.txt').replace('.tiff', '.txt')
    lines = []
    for ann in img_data.get('annotations', []):
        class_id = config.CLASS_MAP.get(ann.get('class', 'individual_tree'), 0)
        polygon = ann.get('segmentation', [])
        if len(polygon) >= 6:
            coords = convert_polygon_to_yolo(polygon, img_w, img_h)
            lines.append(f"{class_id} " + ' '.join(f"{c:.6f}" for c in coords))
    with open(label_path, 'w') as f: f.write('\n'.join(lines))

def train_ensemble(yaml_path):
    from ultralytics import YOLO
    import torch
    
    # Check if GPU is actually available
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"USING DEVICE: {device}")
    
    trained_models = []
    
    for i, model_cfg in enumerate(config.MODELS):
        torch.cuda.empty_cache()
        model = YOLO(model_cfg['arch'])
        
        # Determine augmentation
        aug_params = {
            'mosaic': 1.0, 'mixup': 0.3 if model_cfg['augment'] == 'heavy' else 0.15,
            'copy_paste': 0.5 if model_cfg['augment'] == 'heavy' else 0.3,
            'degrees': 20 if model_cfg['augment'] == 'heavy' else 15,
            'flipud': 0.5, 'fliplr': 0.5
        }
        
        # Training
        model.train(
            data=str(yaml_path),
            epochs=model_cfg['epochs'],
            imgsz=model_cfg['imgsz'],
            batch=model_cfg['batch'],
            device=device,              # FORCE GPU
            project=config.SAVE_DIR,    # SAVE TO DRIVE
            name=model_cfg['name'],
            exist_ok=True,
            amp=True,
            **aug_params
        )
        
        model_path = f"{config.SAVE_DIR}/{model_cfg['name']}/weights/best.pt"
        trained_models.append(model_path)

    print("\nALL MODELS SAVED TO DRIVE SUCCESSFULLY!")

if __name__ == "__main__":
    yaml_path = convert_to_yolo()
    train_ensemble(yaml_path)