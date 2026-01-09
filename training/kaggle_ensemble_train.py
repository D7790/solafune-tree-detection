"""
KAGGLE ENSEMBLE TRAINING - FIXED FOR DDP ISSUES
================================================
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm


class Config:
    DATASET_NAME = "solofune-tree"  # Your dataset name
    
    INPUT_PATH = Path(f"/kaggle/input/{DATASET_NAME}")
    SOURCE_IMAGES = INPUT_PATH / "train_images"
    SOURCE_ANNOTATIONS = INPUT_PATH / "train_annotations_updated.json"
    
    OUTPUT_DIR = Path("/kaggle/working/yolo_dataset")
    SAVE_DIR = "/kaggle/working/runs/ensemble"
    
    MODELS = [
        {"name": "model_1_512", "arch": "yolov8s-seg", "imgsz": 512, "batch": 16, "epochs": 150, "augment": "default"},
        {"name": "model_2_640", "arch": "yolov8s-seg", "imgsz": 640, "batch": 12, "epochs": 150, "augment": "default"},
        {"name": "model_3_heavy_aug", "arch": "yolov8s-seg", "imgsz": 512, "batch": 16, "epochs": 150, "augment": "heavy"}
    ]
    
    TRAIN_RATIO = 0.85
    SEED = 42
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


def process_image(img_data, images_dir, labels_dir):
    filename = img_data['file_name']
    src_path = config.SOURCE_IMAGES / filename
    dst_path = images_dir / filename
    
    if not src_path.exists():
        return False
    
    # USE SYMLINK instead of copy (faster + works with DDP)
    if not dst_path.exists():
        try:
            os.symlink(str(src_path), str(dst_path))
        except:
            import shutil
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
    return True


def convert_to_yolo():
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)
    
    train_images = config.OUTPUT_DIR / "images" / "train"
    val_images = config.OUTPUT_DIR / "images" / "val"
    train_labels = config.OUTPUT_DIR / "labels" / "train"
    val_labels = config.OUTPUT_DIR / "labels" / "val"
    
    for d in [train_images, val_images, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    with open(config.SOURCE_ANNOTATIONS) as f:
        data = json.load(f)
    
    images = data['images']
    random.seed(config.SEED)
    random.shuffle(images)
    
    split_idx = int(len(images) * config.TRAIN_RATIO)
    train_data, val_data = images[:split_idx], images[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    for img_data in tqdm(train_data, desc="Train"):
        process_image(img_data, train_images, train_labels)
    for img_data in tqdm(val_data, desc="Val"):
        process_image(img_data, val_images, val_labels)
    
    # Verify files exist
    train_files = list(train_images.glob('*.tif')) + list(train_images.glob('*.tiff'))
    print(f"Train images found: {len(train_files)}")
    
    if len(train_files) == 0:
        print("ERROR: No images found! Check your dataset path.")
        return None
    
    yaml_path = config.OUTPUT_DIR / "tree_canopy.yaml"
    yaml_content = f"""path: {config.OUTPUT_DIR}
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


def train_ensemble(yaml_path):
    from ultralytics import YOLO
    import torch
    
    if yaml_path is None:
        print("ERROR: Data conversion failed!")
        return
    
    # USE SINGLE GPU to avoid DDP issues
    # DDP has problems with symlinks and worker processes
    device = 0
    print(f"\nUSING DEVICE: {device} (single GPU to avoid DDP issues)")
    
    trained_models = []
    
    for i, model_cfg in enumerate(config.MODELS):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/3: {model_cfg['name']}")
        print(f"{'='*60}")
        
        torch.cuda.empty_cache()
        model = YOLO(model_cfg['arch'])
        
        if model_cfg['augment'] == 'heavy':
            aug = {'mosaic': 1.0, 'mixup': 0.3, 'copy_paste': 0.5, 'degrees': 20, 'scale': 0.7}
        else:
            aug = {'mosaic': 1.0, 'mixup': 0.15, 'copy_paste': 0.3, 'degrees': 15, 'scale': 0.5}
        
        model.train(
            data=str(yaml_path),
            epochs=model_cfg['epochs'],
            imgsz=model_cfg['imgsz'],
            batch=model_cfg['batch'],
            device=device,
            patience=30,
            project=str(config.SAVE_DIR),
            name=model_cfg['name'],
            exist_ok=True,
            amp=True,
            workers=4,
            **aug
        )
        
        model_path = f"{config.SAVE_DIR}/{model_cfg['name']}/weights/best.pt"
        trained_models.append(model_path)
        print(f"Saved: {model_path}")
    
    with open("/kaggle/working/ensemble_models.json", 'w') as f:
        json.dump(trained_models, f)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    yaml_path = convert_to_yolo()
    train_ensemble(yaml_path)
