# Solafune Tree Canopy Detection

A YOLOv8 instance segmentation solution for the Solafune Tree Canopy Detection competition.

## Competition Overview

- **Platform:** [Solafune](https://solafune.com)
- **Task:** Tree canopy detection using satellite imagery
- **Metric:** IoU-based polygon matching
- **Best Score Achieved:** 0.31 IoU

## Project Structure

```
solafune-git/
├── README.md
├── requirements.txt
│
├── data_preparation/
│   └── convert_to_yolo.py          # Convert annotations to YOLO format
│
├── training/
│   ├── train_yolo.py               # Basic YOLO training
│   ├── colab_train_yolo.py         # Colab-optimized training
│   ├── colab_ensemble_train.py     # Ensemble training for Colab
│   ├── kaggle_ensemble_train.py    # Ensemble training for Kaggle
│   └── kaggle_5fold_train.py       # 5-Fold cross validation training
│
├── inference/
│   ├── inference_yolo.py           # Basic YOLO inference
│   ├── colab_inference_yolo.py     # Colab inference
│   ├── colab_ensemble_inference.py # Colab ensemble inference
│   ├── kaggle_ensemble_inference.py# Kaggle ensemble inference  
│   ├── kaggle_5fold_inference.py   # 5-Fold ensemble inference
│   ├── local_ensemble_inference.py # Local inference with downloaded models
│   └── inference_watershed.py      # Watershed post-processing (alternative)
│
├── models/                         # Download from Hugging Face (see below)
│
└── docs/
    └── COLAB_INSTRUCTIONS.md       # Google Colab setup guide
```

## 🤗 Model Weights (Hugging Face)

Models are hosted on Hugging Face Hub:

**Download:** (https://huggingface.co/TheD7/solafune-tree-detection/tree/main))

| Model | Size | Description |
|-------|------|-------------|
| `model_1_512_best.pt` | 23 MB | YOLOv8s-seg @ 512px |
| `model_2_640_best.pt` | 24 MB | YOLOv8s-seg @ 640px |
| `model_3_heavy_aug_best.pt` | 71 MB | YOLOv8s-seg heavy augmentation |

### Download models:
```python
# Option 1: Manual download from Hugging Face
# Option 2: Using huggingface_hub
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="YOUR_USERNAME/solafune-tree-detection", filename="model_1_512_best.pt")
```

## Key Approaches Tried

### 1. YOLOv8 Instance Segmentation
- Used `yolov8s-seg` (small) model
- Trained on 150 annotated images
- Applied strong augmentation (mosaic, mixup, copy-paste)

### 2. Ensemble Approach
- 3 models with different configurations
- Grid-based NMS for deduplication
- Multi-scale TTA at inference

### 3. K-Fold Cross Validation
- 5-fold CV to use all training data
- Each model sees all images across folds

## Results

| Approach | Score |
|----------|-------|
| Single model (no NMS) | 0.16 |
| Ensemble + NMS (conf=0.25) | 0.29 |
| Ensemble + NMS (conf=0.20) | 0.31 |

## Lessons Learned

1. **Data is key:** 150 images is very limited for tree detection
2. **NMS matters:** Remove duplicates from ensemble predictions
3. **Confidence tuning:** Lower confidence (0.20) worked better than higher (0.35)
4. **GPU resources:** Kaggle T4 GPUs are faster than local laptop GPU

## How to Use

### Training on Kaggle
```python
# 1. Upload dataset to Kaggle
# 2. Copy kaggle_ensemble_train.py
# 3. Run training (~3 hours)
run_ensemble_training()
```

### Inference
```python
# Load models and run inference
from ultralytics import YOLO
model = YOLO("models/model_1_512_best.pt")
results = model.predict(image_path, conf=0.20)
```

## Requirements

```
ultralytics>=8.0.0
opencv-python-headless
numpy
tqdm
torch
```

## What Could Improve Score

1. **Pseudo-labeling:** Use predictions on eval images as training data
2. **External data:** Find similar tree detection datasets
3. **Larger models:** yolov8m-seg or yolov8l-seg
4. **More epochs:** Train for 300+ epochs

## Author

Competition attempt: January 2026

---

*Note: This project was discontinued due to resource constraints. Final score: 0.31 IoU (Leaderboard top scores: ~0.55)*
