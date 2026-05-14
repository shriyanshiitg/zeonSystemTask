# Tube Detection and Orientation Estimation

A computer vision system for detecting tube positions and estimating their orientations in overhead RGB imagery. Combines YOLOv8-OBB for detection with classical and neural angle estimation approaches.

## Dataset

- **70 images** (640×480 px) captured from overhead viewpoint
- **371 tube annotations** (3–6 tubes per image)
- Varied backgrounds: desk surfaces, white/black/mixed mats
- Ground truth: bounding box + rotation angle (0–360°)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run scripts in order:

```bash
# 1. Explore dataset statistics
python src/explore.py

# 2. Baseline: Hough circle detection + Sobel angle estimation
python src/baseline.py

# 3. Convert annotations to YOLO OBB format
python src/convert_annotations.py

# 4. Train YOLOv8-OBB detector
python src/train_yolo.py

# 5a. Evaluate YOLO detection performance
python src/evaluate_yolo.py

# 5b. Train ResNet-18 angle head on cropped patches
python src/train_angle_head.py

# 5c. Classical CV angle estimation
python src/classical_angle.py

# 5d. Full pipeline: YOLO + classical angle
python src/evaluate_final.py

# 5e. Combined evaluation
python src/evaluate_combined.py

# 5f. Train multiscale angle heads (64/96/128px crops) — produces final model
python src/train_angle_multiscale.py
```

## Results

| Method | Precision | Recall | F1 | Mean Angle Error | Within 10° | Within 20° |
|--------|-----------|--------|-----|------------------|------------|------------|
| Hough baseline | 0.70 | 0.18 | 0.28 | 101° | — | — |
| YOLOv8-OBB (bbox angle only) | 1.00 | 1.00 | 1.00 | 82° | 4.6% | 13.2% |
| YOLO + ResNet-18 64×64 (leaky split) | 1.00 | 1.00 | 1.00 | 7.24° | — | — |
| YOLO + ResNet-18 96×96 (image-level split) | 1.00 | 1.00 | 1.00 | 4.68° | 93.5% | 100% |

**Final pipeline:** YOLO detection + ResNet-18 96×96 angle head
**Key results:** F1=1.00, Mean Angle Error=4.68°, 93.5% within 10°

> **Data leakage fix:** An initial tube-level split (random tubes, same image in both sets) produced inflated results. Correcting to an image-level split (56 train / 14 val images, zero overlap) revealed the model genuinely generalises with 4.68° mean error on unseen images — not merely memorising training backgrounds. Image-level splitting is essential when training on crops from shared images.

## Directory Structure

```
zeonSystemTask/
├── data/
│   ├── annotations.csv
│   ├── images/
│   └── yolo_dataset/
├── models/
│   └── weights/
│       ├── best.pt              # YOLO detector
│       └── angle_head_best.pth # ResNet-18 angle head (96×96)
├── results/
├── src/
└── report/
```