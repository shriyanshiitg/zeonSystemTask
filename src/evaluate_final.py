"""
Final Evaluation: YOLO Detection + Classical Angle Estimation

This script combines:
1. YOLO detection for tube centers from ./runs/obb/runs/tube_obb/weights/best.pt
2. Classical tab-finding method for angle estimation

Evaluation metrics: Precision, Recall, F1, Mean/Median Angle Error, % within 10°/20°/30°
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import math
from collections import defaultdict

# Configuration
IMAGE_DIR = Path("./images")
ANNOTATIONS_FILE = Path("annotations.csv")
YOLO_WEIGHTS = Path("./runs/obb/runs/tube_obb/weights/best.pt")
OUTPUT_SUMMARY = Path("final_evaluation_summary.txt")
CROP_SIZE = 96
IMAGE_SIZE = (640, 480)
DEVICE = 'cpu'
IOU_THRESHOLD = 30  # Distance threshold in pixels for matching


def crop_patch(image, center_x, center_y, size=96):
    """Crop a square patch centered at (center_x, center_y)."""
    h, w = image.shape[:2]
    half = size // 2

    x1 = int(center_x - half)
    x2 = int(center_x + half)
    y1 = int(center_y - half)
    y2 = int(center_y + half)

    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
        pad_x1 = max(0, -x1)
        pad_x2 = max(0, x2 - w)
        pad_y1 = max(0, -y1)
        pad_y2 = max(0, y2 - h)

        crop = np.zeros((y2 - y1, x2 - x1), dtype=image.dtype)
        if image.ndim == 3:
            crop = np.zeros((y2 - y1, x2 - x1, image.shape[2]), dtype=image.dtype)

        src_x1 = max(0, x1)
        src_x2 = min(w, x2)
        src_y1 = max(0, y1)
        src_y2 = min(h, y2)

        dst_x1 = pad_x1
        dst_x2 = pad_x1 + (src_x2 - src_x1)
        dst_y1 = pad_y1
        dst_y2 = pad_y1 + (src_y2 - src_y1)

        crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    else:
        crop = image[y1:y2, x1:x2].copy()

    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size))

    return crop


def find_tab_direction(crop):
    """Find tab direction using classical CV."""
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    crop_center_x = w / 2
    crop_center_y = h / 2

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 100:
        return 0.0

    (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)

    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(circle_x), int(circle_y)), int(radius), 255, -1)

    circle_region = blurred.copy()
    circle_region_masked = cv2.bitwise_and(circle_region, circle_region, mask=circle_mask)

    # Invert to find dark regions (tab opening)
    inverted = 255 - circle_region_masked

    y_coords, x_coords = np.mgrid[:h, :w]
    weighted_x = x_coords * inverted
    weighted_y = y_coords * inverted

    total_inverted = np.sum(inverted)

    if total_inverted > 0:
        centroid_x = np.sum(weighted_x) / total_inverted
        centroid_y = np.sum(weighted_y) / total_inverted
    else:
        centroid_x = crop_center_x
        centroid_y = crop_center_y

    dx = centroid_x - crop_center_x
    dy = centroid_y - crop_center_y

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def circular_error(pred, gt):
    """Calculate circular error (minimum angular distance)."""
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


def match_detections_to_gt(predictions, ground_truth, iou_threshold=30):
    """
    Match predictions to ground truth based on distance.
    Returns: matched_pairs, false_positives, false_negatives
    """
    matched = []
    gt_matched = set()
    pred_matched = set()

    # For each prediction, find the closest GT
    for pred_idx, pred in enumerate(predictions):
        min_dist = float('inf')
        min_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in gt_matched:
                continue
            dist = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            if dist < min_dist:
                min_dist = dist
                min_gt_idx = gt_idx

        if min_dist <= iou_threshold and min_gt_idx >= 0:
            matched.append({
                'pred_idx': pred_idx,
                'gt_idx': min_gt_idx,
                'pred': pred,
                'gt': ground_truth[min_gt_idx],
                'distance': min_dist
            })
            gt_matched.add(min_gt_idx)
            pred_matched.add(pred_idx)

    # False positives: predictions not matched
    false_positives = [pred for i, pred in enumerate(predictions) if i not in pred_matched]

    # False negatives: GT not matched
    false_negatives = [gt for i, gt in enumerate(ground_truth) if i not in gt_matched]

    return matched, false_positives, false_negatives


def evaluate_full_pipeline():
    """Run the full pipeline: YOLO detection + classical angle estimation."""
    print("Loading YOLO model...")
    model = YOLO(str(YOLO_WEIGHTS))
    model.to(DEVICE)

    print("Loading ground truth annotations...")
    gt_df = pd.read_csv(ANNOTATIONS_FILE)
    print(f"Loaded {len(gt_df)} GT annotations")

    # Group GT by image
    gt_by_image = defaultdict(list)
    for _, row in gt_df.iterrows():
        gt_by_image[row['image']].append({
            'x': row['center_x'],
            'y': row['center_y'],
            'angle': row['angle_deg']
        })

    # Process each image with YOLO
    all_predictions = []
    all_ground_truth = []
    angle_results = []

    image_files = sorted(list(IMAGE_DIR.glob("*-color.png")))
    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        image_name = img_path.name

        # Run YOLO detection
        results = model.predict(
            str(img_path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        # Extract detections
        if len(results) > 0 and results[0].obb is not None and len(results[0].obb) > 0:
            boxes = results[0].obb
            for box in boxes:
                # Get center from OBB (xyxyxyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2

                all_predictions.append({
                    'image': image_name,
                    'x': float(cx),
                    'y': float(cy),
                    'conf': float(box.conf[0])
                })

        # Add ground truth
        if image_name in gt_by_image:
            for gt in gt_by_image[image_name]:
                all_ground_truth.append({
                    'image': image_name,
                    'x': gt['x'],
                    'y': gt['y'],
                    'angle': gt['angle']
                })

    print(f"\nTotal GT objects: {len(all_ground_truth)}")
    print(f"Total YOLO detections (all images): {len(all_predictions)}")

    # Match predictions to GT by image
    predictions_by_image = defaultdict(list)
    for pred in all_predictions:
        predictions_by_image[pred['image']].append(pred)

    ground_truth_by_image = defaultdict(list)
    for gt in all_ground_truth:
        ground_truth_by_image[gt['image']].append(gt)

    # Match within each image
    total_matched = 0
    total_fp = 0
    total_fn = 0

    matched_pairs = []

    for image_name in ground_truth_by_image.keys():
        preds = predictions_by_image.get(image_name, [])
        gts = ground_truth_by_image[image_name]

        matched, fps, fns = match_detections_to_gt(preds, gts, IOU_THRESHOLD)

        total_matched += len(matched)
        total_fp += len(fps)
        total_fn += len(fns)

        matched_pairs.extend(matched)

        # For matched pairs, run classical angle estimation
        for match in matched:
            pred = match['pred']
            gt = match['gt']

            # Load image and crop
            img = cv2.imread(str(IMAGE_DIR / image_name))
            crop = crop_patch(img, pred['x'], pred['y'], CROP_SIZE)

            # Find angle
            pred_angle = find_tab_direction(crop)
            gt_angle = gt['angle']

            angle_error = circular_error(pred_angle, gt_angle)

            angle_results.append({
                'image': image_name,
                'pred_x': pred['x'],
                'pred_y': pred['y'],
                'gt_x': gt['x'],
                'gt_y': gt['y'],
                'gt_angle': gt_angle,
                'pred_angle': pred_angle,
                'error': angle_error
            })

    # Calculate detection metrics
    precision = total_matched / (total_matched + total_fp) if (total_matched + total_fp) > 0 else 0
    recall = total_matched / (total_matched + total_fn) if (total_matched + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetection Results:")
    print(f"  True Positives (matched): {total_matched}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Calculate angle metrics (only for matched detections)
    if angle_results:
        errors = [r['error'] for r in angle_results]

        mean_error = np.mean(errors)
        median_error = np.median(errors)

        within_10 = np.sum(np.array(errors) <= 10) / len(errors) * 100
        within_20 = np.sum(np.array(errors) <= 20) / len(errors) * 100
        within_30 = np.sum(np.array(errors) <= 30) / len(errors) * 100

        print(f"\nAngle Estimation Results (on {len(errors)} matched detections):")
        print(f"  Mean angle error: {mean_error:.2f}°")
        print(f"  Median angle error: {median_error:.2f}°")
        print(f"  % within 10°: {within_10:.1f}%")
        print(f"  % within 20°: {within_20:.1f}%")
        print(f"  % within 30°: {within_30:.1f}%")
    else:
        mean_error = median_error = within_10 = within_20 = within_30 = 0
        print("\nNo angle results (no matched detections)")

    # Save summary
    summary = f"""FINAL EVALUATION SUMMARY
========================

Detection Metrics:
  True Positives (matched): {total_matched}
  False Positives: {total_fp}
  False Negatives: {total_fn}
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1 Score: {f1:.4f}

Angle Estimation Metrics (on matched detections):
  Total matched: {len(angle_results) if angle_results else 0}
  Mean angle error: {mean_error:.2f}°
  Median angle error: {median_error:.2f}°
  % within 10°: {within_10:.1f}%
  % within 20°: {within_20:.1f}%
  % within 30°: {within_30:.1f}%

Configuration:
  YOLO weights: {YOLO_WEIGHTS}
  Image size: {IMAGE_SIZE}
  Device: {DEVICE}
  Classical crop size: {CROP_SIZE}x{CROP_SIZE}
  IOU matching threshold: {IOU_THRESHOLD} pixels
"""

    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write(summary)

    print(f"\nSaved summary to {OUTPUT_SUMMARY}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_error': mean_error,
        'median_error': median_error,
        'within_10': within_10,
        'within_20': within_20,
        'within_30': within_30
    }


if __name__ == "__main__":
    evaluate_full_pipeline()