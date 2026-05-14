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
```

## Results

| Method | Precision | Recall | F1 | Mean Angle Error |
|--------|-----------|--------|-----|------------------|
| Hough baseline | 0.70 | 0.18 | 0.28 | 101° |
| YOLOv8-OBB (angle from OBB) | 1.00 | 1.00 | 1.00 | 82° |
| YOLO + ResNet-18 angle head | 1.00 | 1.00 | 1.00 | 88° |
| YOLO + classical CV | 1.00 | 1.00 | 1.00 | 96° |

Detection is solved (F1=1.00). Angle estimation remains challenging due to 180° ambiguity in the rotationally symmetric tube lid.

## Directory Structure

```
zeonSystemTask/
├── data/
│   ├── annotations.csv
│   ├── images/
│   └── yolo_dataset/
├── models/
│   ├── weights/best.pt
│   └── angle_head.pth
├── results/
├── src/
└── report/
```