import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load ground truth
gt_df = pd.read_csv('annotations.csv')

# Get all images
image_files = sorted([f for f in os.listdir('./images') if f.endswith('.png')])
print(f"Found {len(image_files)} images")

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Parameters
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 0.7
MIN_RADIUS = 15
MAX_RADIUS = 50
ANGLE_PATCH_SIZE = 64

def estimate_angle(img_gray, cx, cy, patch_size=64):
    """Estimate angle using Sobel gradients and circular mean."""
    half = patch_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_gray.shape[1], cx + half)
    y2 = min(img_gray.shape[0], cy + half)

    patch = img_gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    # Resize to fixed size if needed
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))

    # Compute Sobel gradients
    sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and phase
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    phase = cv2.phase(sobel_x, -sobel_y, angleInDegrees=True)

    # Weight by magnitude and compute circular mean
    weights = magnitude.flatten()
    angles = phase.flatten()

    if weights.sum() == 0:
        return 0.0

    # Circular mean
    sin_sum = np.sum(weights * np.sin(np.radians(angles)))
    cos_sum = np.sum(weights * np.cos(np.radians(angles)))
    mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

    return mean_angle

def circular_distance(a1, a2):
    """Compute circular distance between two angles in degrees."""
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

# Storage for results
all_predictions = []
all_gt = []

print("Processing images...")
for img_name in image_files:
    img_path = f"./images/{img_name}"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load {img_name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT_ALT,
        dp=1.5, minDist=30,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
    )

    img_gt = gt_df[gt_df['image'] == img_name]
    predictions = []

    if circles is not None:
        for circle in circles[0]:
            cx, cy, r = circle
            cx, cy, r = int(cx), int(cy), int(r)
            angle = estimate_angle(gray, cx, cy, ANGLE_PATCH_SIZE)
            predictions.append({'cx': cx, 'cy': cy, 'angle': angle, 'r': r})

    # Store results
    for pred in predictions:
        all_predictions.append({
            'image': img_name,
            'pred_cx': pred['cx'],
            'pred_cy': pred['cy'],
            'pred_angle': pred['angle']
        })

    for _, row in img_gt.iterrows():
        all_gt.append({
            'image': img_name,
            'gt_cx': row['center_x'],
            'gt_cy': row['center_y'],
            'gt_angle': row['angle_deg']
        })

    print(f"  {img_name}: {len(predictions)} predictions, {len(img_gt)} GT tubes")

# Match predictions to GT
results = []
matched_gt_indices = set()

for pred_idx, pred in enumerate(all_predictions):
    best_dist = float('inf')
    best_gt_idx = -1

    for gt_idx, gt in enumerate(all_gt):
        if gt['image'] != pred['image']:
            continue
        dist = np.sqrt((pred['pred_cx'] - gt['gt_cx'])**2 + (pred['pred_cy'] - gt['gt_cy'])**2)
        if dist < best_dist and dist <= 20:
            best_dist = dist
            best_gt_idx = gt_idx

    if best_gt_idx >= 0:
        gt = all_gt[best_gt_idx]
        angle_err = circular_distance(pred['pred_angle'], gt['gt_angle'])
        results.append({
            'image': pred['image'],
            'pred_cx': pred['pred_cx'],
            'pred_cy': pred['pred_cy'],
            'pred_angle': pred['pred_angle'],
            'matched_gt_cx': gt['gt_cx'],
            'matched_gt_cy': gt['gt_cy'],
            'gt_angle': gt['gt_angle'],
            'angle_error': angle_err,
            'is_tp': True
        })
        matched_gt_indices.add((pred['image'], best_gt_idx))
    else:
        results.append({
            'image': pred['image'],
            'pred_cx': pred['pred_cx'],
            'pred_cy': pred['pred_cy'],
            'pred_angle': pred['pred_angle'],
            'matched_gt_cx': None,
            'matched_gt_cy': None,
            'gt_angle': None,
            'angle_error': None,
            'is_tp': False
        })

# Add unmatched GT as FNs
for gt_idx, gt in enumerate(all_gt):
    if (gt['image'], gt_idx) not in matched_gt_indices:
        results.append({
            'image': gt['image'],
            'pred_cx': None,
            'pred_cy': None,
            'pred_angle': None,
            'matched_gt_cx': gt['gt_cx'],
            'matched_gt_cy': gt['gt_cy'],
            'gt_angle': gt['gt_angle'],
            'angle_error': None,
            'is_tp': False
        })

results_df = pd.DataFrame(results)
results_df.to_csv('baseline_results.csv', index=False)
print(f"\nSaved results to baseline_results.csv")

# Compute metrics
tp = results_df['is_tp'].sum()
fp = len(results_df) - tp - sum(1 for r in results if r['pred_cx'] is None and r['matched_gt_cx'] is None)
fn = sum(1 for r in results if r['pred_cx'] is None and r['matched_gt_cx'] is not None)

# Recalculate properly
total_preds = len(all_predictions)
total_gt = len(all_gt)
tp_count = sum(1 for r in results if r['is_tp'])
fp_count = total_preds - tp_count
fn_count = total_gt - tp_count

precision = tp_count / total_preds if total_preds > 0 else 0
recall = tp_count / total_gt if total_gt > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Angle errors for TPs
angle_errors = results_df[results_df['is_tp']]['angle_error'].dropna()
mean_angle_error = angle_errors.mean() if len(angle_errors) > 0 else 0
median_angle_error = angle_errors.median() if len(angle_errors) > 0 else 0

print(f"\n=== Metrics ===")
print(f"Total predictions: {total_preds}")
print(f"Total GT: {total_gt}")
print(f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Angle Error: {mean_angle_error:.2f} deg")
print(f"Median Angle Error: {median_angle_error:.2f} deg")

# Visualization: 5 sample images
sample_images = image_files[:5]
cols = 5
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
if cols == 1:
    axes = [axes]

for idx, img_name in enumerate(sample_images):
    img = cv2.imread(f"./images/{img_name}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get predictions for this image
    img_preds = [r for r in results if r['image'] == img_name and r['is_tp']]
    # Get GT for this image
    img_gt = gt_df[gt_df['image'] == img_name]

    # Draw GT (blue circles)
    for _, row in img_gt.iterrows():
        cx, cy = int(row['center_x']), int(row['center_y'])
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), 2)

    # Draw predictions (green circles, red direction lines)
    for p in img_preds:
        cx, cy = int(p['pred_cx']), int(p['pred_cy'])
        cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)
        angle = p['pred_angle']
        rad = np.radians(angle)
        dx = 30 * np.cos(rad)
        dy = 30 * np.sin(rad)
        end_x = int(cx + dx)
        end_y = int(cy - dy)
        cv2.line(img, (cx, cy), (end_x, end_y), (0, 0, 255), 2)

    axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[idx].axis('off')
    axes[idx].set_title(img_name[:15])

plt.tight_layout()
plt.savefig('baseline_viz.png', dpi=150)
print("\nSaved visualization to baseline_viz.png")