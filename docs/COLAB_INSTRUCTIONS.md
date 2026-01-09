# YOLO Training on Google Colab - Step by Step Guide

## 🎯 Target: 0.60+ IoU on Solafune

---

## Step 1: Upload Data to Google Drive

Create this folder structure in your Google Drive:
```
MyDrive/
└── solafune_tree/
    ├── train_images/           # Copy all 150 .tif files
    ├── train_annotations.json  # The annotation file
    └── evaluation_images/      # Copy all 150 evaluation .tif files
```

---

## Step 2: Create New Colab Notebook

Go to https://colab.research.google.com and create a new notebook.

**IMPORTANT:** Go to `Runtime → Change runtime type → Select GPU`
- Best: A100 (if available)
- Good: V100 or T4

---

## Step 3: Run These Cells

### Cell 1: Install & Mount Drive
```python
!pip install ultralytics opencv-python-headless tqdm albumentations

from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Copy Training Script
```python
# Upload colab_train_yolo.py to Colab, or copy-paste the content here
# The script is in your Solafune folder
```

### Cell 3: Run Training
```python
# Import and run
exec(open('/content/colab_train_yolo.py').read())
```

OR copy the entire content of `colab_train_yolo.py` into a cell and run it.

---

## Step 4: After Training - Run Inference

### Cell 4: Copy Inference Script
```python
# Upload colab_inference_yolo.py or copy-paste content
exec(open('/content/colab_inference_yolo.py').read())
```

### Cell 5: Download Submission
```python
from google.colab import files
files.download('/content/submission_yolo_optimized.json')
```

---

## 📊 Optimizations Applied

| Setting | Your Local | Colab Optimized |
|---------|------------|-----------------|
| Model | yolov8n-seg | **yolov8m-seg** |
| Image Size | 512 | **1024** |
| Epochs | 150 | **300** |
| Batch Size | 4 | 4-16 (auto) |
| Copy-Paste Aug | No | **Yes (0.3)** |
| Multi-Scale | No | **Yes (0.5)** |
| TTA | No | **Yes** |
| Confidence | 0.25 | **0.15** |

---

## 🚀 Expected Results

| Configuration | Expected IoU |
|---------------|--------------|
| Your local (nano, 512px) | 0.20 |
| **Colab (medium, 1024px)** | **0.40-0.50** |
| Colab (large, 1024px) + fine-tune | **0.55-0.65** |

---

## ⚠️ Tips for Better Scores

1. **If OOM error**: Reduce BATCH_SIZE to 2
2. **For even better scores**: Change MODEL to "yolov8l-seg" (large)
3. **Training time**: ~2-4 hours for 300 epochs on T4
4. **Save model to Drive**: The script does this automatically

---

## Files to Upload to Colab

1. `colab_train_yolo.py` - Training script
2. `colab_inference_yolo.py` - Inference script
3. Your dataset in Drive (images + annotations)
