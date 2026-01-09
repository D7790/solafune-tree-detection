"""
YOLO v8 Instance Segmentation Training for Tree Canopy Detection.

Usage:
    python train_yolo.py                    # Full training
    python train_yolo.py --epochs 10        # Quick test
    python train_yolo.py --model yolov8m-seg  # Use medium model
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


# ==================== CONFIG ====================
class Config:
    DATASET_YAML = Path("datasets/yolo_tree/tree_canopy.yaml")
    OUTPUT_DIR = Path("runs/segment")
    
    # Model options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
    DEFAULT_MODEL = "yolov8n-seg"  # Nano model - smallest, fastest
    
    EPOCHS = 150
    IMAGE_SIZE = 512  # Reduced from 640 to save memory
    BATCH_SIZE = 4    # Reduced from 8 to avoid OOM
    
    PATIENCE = 30     # Early stopping patience
    
    # Data augmentation
    AUG_MOSAIC = 1.0
    AUG_MIXUP = 0.1
    AUG_COPY_PASTE = 0.1
    AUG_FLIPUD = 0.5
    AUG_FLIPLR = 0.5
    
    DEVICE = 0  # GPU


config = Config()


def train(args):
    print("=" * 60)
    print("YOLO v8 INSTANCE SEGMENTATION - TREE CANOPY")
    print("=" * 60)
    
    # Check dataset exists
    if not config.DATASET_YAML.exists():
        print(f"ERROR: Dataset not found at {config.DATASET_YAML}")
        print("Run 'python convert_to_yolo.py' first!")
        return
    
    # Load model
    model_name = args.model if args.model else config.DEFAULT_MODEL
    print(f"\nModel: {model_name}")
    print(f"Epochs: {args.epochs if args.epochs else config.EPOCHS}")
    print(f"Image size: {args.imgsz if args.imgsz else config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    # Initialize model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=str(config.DATASET_YAML),
        epochs=args.epochs if args.epochs else config.EPOCHS,
        imgsz=args.imgsz if args.imgsz else config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        device=config.DEVICE,
        
        # Augmentation
        mosaic=config.AUG_MOSAIC,
        mixup=config.AUG_MIXUP,
        copy_paste=config.AUG_COPY_PASTE,
        flipud=config.AUG_FLIPUD,
        fliplr=config.AUG_FLIPLR,
        
        # Output
        project=str(config.OUTPUT_DIR),
        name="tree_canopy",
        exist_ok=True,
        
        # Performance
        amp=True,  # Mixed precision
        workers=4,
        
        # Saving
        save=True,
        save_period=20,  # Save every 20 epochs
        
        # Logging
        verbose=True,
        plots=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model: {config.OUTPUT_DIR}/tree_canopy/weights/best.pt")
    print(f"Last model: {config.OUTPUT_DIR}/tree_canopy/weights/last.pt")
    print(f"Results: {config.OUTPUT_DIR}/tree_canopy/")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for tree canopy detection')
    parser.add_argument('--model', type=str, help='Model name (e.g., yolov8s-seg)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, help='Image size')
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
