"""
Classical Computer Vision Angle Estimator for Tube Detection

This script implements a classical CV approach to find the tab direction on tube lids
without needing a trained neural network.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

# Configuration
IMAGE_DIR = Path("./images")
ANNOTATIONS_FILE = Path("annotations.csv")
OUTPUT_FILE = Path("classical_angle_results.csv")
CROP_SIZE = 96


def crop_patch(image, center_x, center_y, size=96):
    """Crop a square patch centered at (center_x, center_y)."""
    h, w = image.shape[:2]
    half = size // 2

    # Calculate crop bounds
    x1 = int(center_x - half)
    x2 = int(center_x + half)
    y1 = int(center_y - half)
    y2 = int(center_y + half)

    # Handle boundaries
    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
        # Pad with edge values if crop goes outside image
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

    # Ensure exactly size x size
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size))

    return crop


def find_tab_direction(crop):
    """
    Find the tab direction using classical CV.

    The tab appears as a dark region (opening) on the lid circle.
    Steps:
    1. Convert to grayscale and apply Gaussian blur
    2. Use Otsu thresholding to find lid circle
    3. Find the intensity-weighted centroid of the DARKEST region within the circle
    4. The tab direction is from crop center to this dark centroid
    """
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    h, w = gray.shape
    crop_center_x = w / 2
    crop_center_y = h / 2

    # Apply Gaussian blur (5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding to find lid circle
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours to identify the lid circle
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Find the largest contour (likely the lid)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 100:
        return 0.0

    # Get bounding circle
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)

    # Create a circular mask for the lid region
    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(circle_x), int(circle_y)), int(radius), 255, -1)

    # Get the intensity values within the circle
    circle_region = blurred.copy()
    circle_region_masked = cv2.bitwise_and(circle_region, circle_region, mask=circle_mask)

    # Invert intensity to find dark regions
    inverted = 255 - circle_region_masked

    # Calculate intensity-weighted centroid of DARK regions
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

    # Calculate angle from crop center to dark centroid
    dx = centroid_x - crop_center_x
    dy = centroid_y - crop_center_y

    # Use atan2 to get angle (returns -pi to pi)
    angle_rad = math.atan2(dy, dx)

    # Convert to degrees (0-360)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def circular_error(pred, gt):
    """Calculate circular error (minimum angular distance)."""
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


def main():
    """Main function to run classical angle estimation on GT centers."""
    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_FILE)
    print(f"Loaded {len(df)} annotations")

    results = []

    for idx, row in df.iterrows():
        image_name = row['image']
        center_x = row['center_x']
        center_y = row['center_y']
        gt_angle = row['angle_deg']

        # Load image
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue

        # Crop patch
        crop = crop_patch(image, center_x, center_y, CROP_SIZE)

        # Find tab direction
        pred_angle = find_tab_direction(crop)

        # Calculate error
        error = circular_error(pred_angle, gt_angle)

        results.append({
            'image': image_name,
            'center_x': center_x,
            'center_y': center_y,
            'gt_angle': gt_angle,
            'pred_angle': pred_angle,
            'error': error
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")

    # Calculate metrics
    errors = results_df['error'].values

    mean_error = np.mean(errors)
    median_error = np.median(errors)

    within_10 = np.sum(errors <= 10) / len(errors) * 100
    within_20 = np.sum(errors <= 20) / len(errors) * 100
    within_30 = np.sum(errors <= 30) / len(errors) * 100

    print("\n" + "=" * 50)
    print("CLASSICAL ANGLE ESTIMATION RESULTS (GT Centers)")
    print("=" * 50)
    print(f"Total samples: {len(errors)}")
    print(f"Mean angle error: {mean_error:.2f}°")
    print(f"Median angle error: {median_error:.2f}°")
    print(f"% within 10°: {within_10:.1f}%")
    print(f"% within 20°: {within_20:.1f}%")
    print(f"% within 30°: {within_30:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()