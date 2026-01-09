"""
WATERSHED INFERENCE - Split merged tree predictions into individual crowns
This addresses the root cause: Solafune uses polygon-based matching, not pixel IoU.

Key insight:
- GT has many small individual trees (e.g., 523 per image)
- Our model predicts merged blobs (e.g., 97 per image)
- We need to SPLIT these blobs into individual trees

Solution: Watershed segmentation + Distance transform
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ==================== CONFIG ====================
class Config:
    DATA_DIR = Path("dataset")
    TEST_IMAGES_DIR = DATA_DIR / "evaluation_images"
    TRAIN_ANNOTATIONS = DATA_DIR / "train_annotations.json"
    
    # MODEL OPTIONS: \"final_boss\", \"anti_overfit\"
    MODEL_NAME = "final_boss"  # anti_overfit has incompatible architecture
    MODEL_PATH = Path(f"models/{MODEL_NAME}/best_model.pth")
    SUBMISSION_PATH = Path(f"submissions/submission_watershed_{MODEL_NAME}.json")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMAGE_SIZE = 512
    NUM_CLASSES = 3
    
    # Watershed parameters - TUNED for tree crowns
    MIN_DISTANCE = 8  # Minimum distance between tree centers
    MIN_TREE_AREA = 30  # Minimum individual tree area
    MAX_TREE_AREA = 5000  # Maximum (split if larger)
    
    # Polygon settings
    EPSILON_FACTOR = 0.005
    MAX_VERTICES = 10


config = Config()


# ==================== MODEL ====================
def load_model():
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
    cfg = checkpoint.get('config', {})
    enc = cfg.get('encoder', 'efficientnet-b4')
    img_size = cfg.get('image_size', config.IMAGE_SIZE)
    
    model = smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, classes=config.NUM_CLASSES).to(config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model: {config.MODEL_NAME}")
        print(f"Encoder: {enc}, Image size: {img_size}")
        print(f"IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, img_size


def get_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ==================== WATERSHED SEGMENTATION ====================
def watershed_split(binary_mask, min_distance=8, min_area=30):
    """
    Use watershed to split merged tree regions into individual trees.
    Non-recursive version to avoid stack overflow.
    
    Args:
        binary_mask: Binary mask of trees (0 or 1)
        min_distance: Minimum distance between tree centers
        min_area: Minimum tree area (filter small noise)
        
    Returns:
        labeled_mask: Each unique value is a different tree instance
    """
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)
    
    # Distance transform - find distance from each pixel to background
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find local maxima (tree centers)
    try:
        local_maxi = peak_local_max(
            distance, 
            min_distance=min_distance,
            labels=binary_mask.astype(np.uint8),
            exclude_border=False
        )
    except:
        # Fallback: just label connected components
        return ndimage.label(binary_mask)[0]
    
    if len(local_maxi) == 0:
        return ndimage.label(binary_mask)[0]
    
    # Create markers for watershed
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxi, start=1):
        markers[y, x] = i
    
    # Apply watershed
    labeled = watershed(-distance, markers, mask=binary_mask.astype(bool))
    
    # Filter by minimum area
    final_labeled = np.zeros_like(labeled)
    new_label = 1
    
    for region_id in range(1, labeled.max() + 1):
        region_mask = (labeled == region_id)
        area = region_mask.sum()
        
        if area >= min_area:
            final_labeled[region_mask] = new_label
            new_label += 1
    
    return final_labeled


def instance_to_polygons(labeled_mask, epsilon_factor=0.005, max_vertices=10):
    """Convert labeled instance mask to list of polygons."""
    polygons = []
    
    for region_id in range(1, labeled_mask.max() + 1):
        region_mask = (labeled_mask == region_id).astype(np.uint8)
        
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Simplify
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Additional simplification if too many vertices
            while len(approx) > max_vertices and epsilon < 0.1 * cv2.arcLength(contour, True):
                epsilon *= 1.5
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                continue
            
            # Convert to flat coords (FLOAT format like training data)
            flat_coords = []
            for pt in approx:
                flat_coords.extend([float(pt[0][0]), float(pt[0][1])])
            
            if len(flat_coords) >= 6:
                polygons.append(flat_coords)
    
    return polygons


# ==================== MAIN INFERENCE ====================
def process_prediction_watershed(pred_mask, original_size):
    """Process prediction mask using watershed to get individual tree polygons."""
    h, w = original_size
    
    # Resize to original
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    annotations = []
    
    # Process each class
    for class_id, class_name in [(1, 'individual_tree'), (2, 'group_of_trees')]:
        class_mask = (pred_mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
        
        # Apply watershed to split merged regions
        if class_id == 1:  # Individual trees - split aggressively
            labeled = watershed_split(
                class_mask,
                min_distance=config.MIN_DISTANCE,
                min_area=config.MIN_TREE_AREA
            )
        else:  # Groups - split less aggressively
            labeled = watershed_split(
                class_mask,
                min_distance=config.MIN_DISTANCE * 2,
                min_area=config.MIN_TREE_AREA * 2
            )
        
        # Convert to polygons
        polygons = instance_to_polygons(
            labeled,
            epsilon_factor=config.EPSILON_FACTOR,
            max_vertices=config.MAX_VERTICES
        )
        
        for polygon in polygons:
            annotations.append({
                'class': class_name,
                'confidence_score': 1.0,
                'segmentation': polygon
            })
    
    return annotations


def run_inference():
    print("\n" + "=" * 60)
    print("WATERSHED INFERENCE - Individual Tree Detection")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Min tree distance: {config.MIN_DISTANCE}px")
    print(f"Min tree area: {config.MIN_TREE_AREA}px")
    print("=" * 60 + "\n")
    
    model, image_size = load_model()
    transform = get_transform(image_size)
    
    # Get test images
    test_images = list(config.TEST_IMAGES_DIR.glob('*.tif')) + list(config.TEST_IMAGES_DIR.glob('*.tiff'))
    print(f"Processing {len(test_images)} images...\n")
    
    submission = {'images': []}
    total_polygons = 0
    
    for img_path in tqdm(test_images, desc='Inference'):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Predict
        preprocessed = transform(image=image_rgb)['image']
        input_tensor = preprocessed.unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_mask = np.argmax(pred, axis=0)
        
        # Get annotations with watershed
        annotations = process_prediction_watershed(pred_mask, (orig_h, orig_w))
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
        
        # Scene type
        scene_type = "agriculture_plantation" if cm_resolution == 10 else "mixed"
        
        submission['images'].append({
            'file_name': filename,
            'width': orig_w,
            'height': orig_h,
            'cm_resolution': cm_resolution,
            'scene_type': scene_type,
            'annotations': annotations
        })
    
    # Save
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
    print("=" * 60)
    
    # Compare with expected
    print("\nCOMPARISON:")
    print("  Before watershed: ~100 polygons/image")
    print(f"  After watershed:  ~{avg_per_image:.0f} polygons/image")
    print("  GT typical:       ~200-500 polygons/image")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_inference()
