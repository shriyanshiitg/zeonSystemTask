"""
Experiment 2: Test-Time Augmentation on Angle Head

Reduces the 180° flip problem by applying multiple augmentations at test time
and aggregating predictions using circular mean.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math
from scipy import stats

# Configuration
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"
MODEL_WEIGHTS = Path("models/angle_head.pth")
OUTPUT_DIR = Path("results/experiment02_tta_angle")
OUTPUT_SUMMARY = OUTPUT_DIR / "evaluation_summary.txt"

DEVICE = 'cpu'
CROP_SIZE = 64


class AngleHead(nn.Module):
    """ResNet-18 based angle regression head."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)


def circular_mean(angles):
    """Compute circular mean of angles in degrees."""
    if not angles:
        return 0.0
    rads = [math.radians(a) for a in angles]
    mean_sin = np.mean([math.sin(r) for r in rads])
    mean_cos = np.mean([math.cos(r) for r in rads])
    mean_rad = math.atan2(mean_sin, mean_cos)
    mean_deg = math.degrees(mean_rad) % 360
    return mean_deg


def circular_error(pred, gt):
    """Calculate circular error (minimum angular distance)."""
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


def apply_rotation(image, angle):
    """Rotate image by angle degrees counter-clockwise."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def adjust_angle_for_rotation(pred_angle, rotation_deg):
    """Adjust predicted angle back after rotation."""
    return (pred_angle - rotation_deg) % 360


def adjust_angle_for_hflip(pred_angle):
    """Adjust predicted angle after horizontal flip."""
    return (180 - pred_angle) % 360


def adjust_angle_for_vflip(pred_angle):
    """Adjust predicted angle after vertical flip."""
    return (360 - pred_angle) % 360


def get_transform():
    """Get image transform for inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def extract_crop(image, center_x, center_y, crop_size=CROP_SIZE):
    """Extract centered crop from image."""
    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = int(center_x) - half
    y1 = int(center_y) - half
    x2 = int(center_x) + half
    y2 = int(center_y) + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=0)

    return crop


def predict_angle(model, crop, transform):
    """Predict angle from a single crop."""
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_tensor = transform(crop_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(crop_tensor)
        pred_sin = output[0, 0].item()
        pred_cos = output[0, 1].item()
        angle = math.degrees(math.atan2(pred_sin, pred_cos)) % 360

    return angle


def tta_predict(model, crop, transform):
    """Test-time augmentation prediction with 10 augmentations."""
    predictions = []

    # 1. Original
    angle = predict_angle(model, crop, transform)
    predictions.append(angle)

    # 2-8. Rotations: 45, 90, 135, 180, 225, 270, 315
    rotations = [45, 90, 135, 180, 225, 270, 315]
    for rot in rotations:
        rotated = apply_rotation(crop, rot)
        angle = predict_angle(model, rotated, transform)
        adjusted = adjust_angle_for_rotation(angle, rot)
        predictions.append(adjusted)

    # 9. Horizontal flip
    hflipped = cv2.flip(crop, 1)
    angle = predict_angle(model, hflipped, transform)
    adjusted = adjust_angle_for_hflip(angle)
    predictions.append(adjusted)

    # 10. Vertical flip
    vflipped = cv2.flip(crop, 0)
    angle = predict_angle(model, vflipped, transform)
    adjusted = adjust_angle_for_vflip(angle)
    predictions.append(adjusted)

    # Circular mean aggregation
    final_angle = circular_mean(predictions)
    return final_angle, predictions


def no_tta_predict(model, crop, transform):
    """Standard prediction without TTA."""
    angle = predict_angle(model, crop, transform)
    return angle


def main():
    """Main evaluation function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = AngleHead()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded weights from {MODEL_WEIGHTS}")

    # Load annotations
    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_FILE)

    # Use validation set (same 20% split as training)
    torch.manual_seed(42)
    indices = torch.randperm(len(df))
    val_size = len(df) - int(0.8 * len(df))
    val_indices = indices[-val_size:].tolist()
    val_df = df.iloc[val_indices].reset_index(drop=True)
    print(f"Validation samples: {len(val_df)}")

    transform = get_transform()

    # Evaluate with TTA
    print("\nEvaluating with TTA (10 augmentations)...")
    tta_errors = []
    tta_results = []

    for idx, row in val_df.iterrows():
        image_path = IMAGE_DIR / row['image']
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        crop = extract_crop(image, row['center_x'], row['center_y'])
        gt_angle = row['angle_deg']

        pred_angle, all_preds = tta_predict(model, crop, transform)
        error = circular_error(pred_angle, gt_angle)
        tta_errors.append(error)
        tta_results.append({
            'image': row['image'],
            'gt_angle': gt_angle,
            'pred_angle': pred_angle,
            'error': error
        })

    # Evaluate without TTA for comparison
    print("Evaluating without TTA...")
    no_tta_errors = []

    for idx, row in val_df.iterrows():
        image_path = IMAGE_DIR / row['image']
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        crop = extract_crop(image, row['center_x'], row['center_y'])
        gt_angle = row['angle_deg']

        pred_angle = no_tta_predict(model, crop, transform)
        error = circular_error(pred_angle, gt_angle)
        no_tta_errors.append(error)

    # Compute metrics
    tta_mean = np.mean(tta_errors)
    tta_median = np.median(tta_errors)
    tta_within_10 = np.sum(np.array(tta_errors) <= 10) / len(tta_errors) * 100
    tta_within_20 = np.sum(np.array(tta_errors) <= 20) / len(tta_errors) * 100
    tta_within_30 = np.sum(np.array(tta_errors) <= 30) / len(tta_errors) * 100

    no_tta_mean = np.mean(no_tta_errors)
    no_tta_median = np.median(no_tta_errors)
    no_tta_within_10 = np.sum(np.array(no_tta_errors) <= 10) / len(no_tta_errors) * 100
    no_tta_within_20 = np.sum(np.array(no_tta_errors) <= 20) / len(no_tta_errors) * 100
    no_tta_within_30 = np.sum(np.array(no_tta_errors) <= 30) / len(no_tta_errors) * 100

    # Print results
    print("\n" + "=" * 60)
    print("TEST-TIME AUGMENTATION EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Without TTA ---")
    print(f"  Mean angle error: {no_tta_mean:.2f}°")
    print(f"  Median angle error: {no_tta_median:.2f}°")
    print(f"  % within 10°: {no_tta_within_10:.1f}%")
    print(f"  % within 20°: {no_tta_within_20:.1f}%")
    print(f"  % within 30°: {no_tta_within_30:.1f}%")

    print("\n--- With TTA (10 augmentations) ---")
    print(f"  Mean angle error: {tta_mean:.2f}°")
    print(f"  Median angle error: {tta_median:.2f}°")
    print(f"  % within 10°: {tta_within_10:.1f}%")
    print(f"  % within 20°: {tta_within_20:.1f}%")
    print(f"  % within 30°: {tta_within_30:.1f}%")

    improvement_mean = no_tta_mean - tta_mean
    print(f"\n  Improvement in mean error: {improvement_mean:.2f}°")

    # Save summary
    summary = f"""TEST-TIME AUGMENTATION EVALUATION SUMMARY
============================================
Experiment: 02 - TTA on Angle Head

Method: 10-way test-time augmentation with circular mean aggregation
Augmentations:
  - Original
  - Rotations: 45°, 90°, 135°, 180°, 225°, 270°, 315° (angle adjusted on prediction)
  - Horizontal flip (angle adjusted: 180° - angle)
  - Vertical flip (angle adjusted: 360° - angle)

Aggregation: Circular mean of all predictions

Validation Samples: {len(val_df)}

--- Without TTA ---
  Mean angle error: {no_tta_mean:.2f}°
  Median angle error: {no_tta_median:.2f}°
  % within 10°: {no_tta_within_10:.1f}%
  % within 20°: {no_tta_within_20:.1f}%
  % within 30°: {no_tta_within_30:.1f}%

--- With TTA ---
  Mean angle error: {tta_mean:.2f}°
  Median angle error: {tta_median:.2f}°
  % within 10°: {tta_within_10:.1f}%
  % within 20°: {tta_within_20:.1f}%
  % within 30°: {tta_within_30:.1f}%

Improvement:
  Mean error reduction: {improvement_mean:.2f}°

Notes:
  - Uses ResNet-18 angle head from models/angle_head.pth
  - Crop size: {CROP_SIZE}x{CROP_SIZE}
  - Device: {DEVICE}
  - TTA helps resolve 180° ambiguity by considering multiple viewpoints
"""

    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write(summary)

    print(f"\nSaved summary to {OUTPUT_SUMMARY}")

    return {
        'tta_mean': tta_mean,
        'tta_median': tta_median,
        'tta_within_10': tta_within_10,
        'tta_within_20': tta_within_20,
        'tta_within_30': tta_within_30,
        'no_tta_mean': no_tta_mean,
        'improvement': improvement_mean
    }


if __name__ == "__main__":
    main()