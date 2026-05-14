import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load annotations
df = pd.read_csv('annotations.csv')
print("=== First 20 tubes ===")
print(f"{'image':<25} {'center_x':>8} {'center_y':>8} {'angle_deg':>10} {'bbox_rot':>10} {'diff':>8}")
print("-" * 75)

for i in range(min(20, len(df))):
    row = df.iloc[i]
    diff = abs(row['angle_deg'] - row['bbox_rotation']) % 360
    if diff > 180:
        diff = 360 - diff
    print(f"{row['image']:<25} {row['center_x']:>8.1f} {row['center_y']:>8.1f} {row['angle_deg']:>10.1f} {row['bbox_rotation']:>10.1f} {diff:>8.1f}")

# Histogram of angle_deg
print("\n=== Angle Distribution (8 directions) ===")
angles = df['angle_deg'].values
bins = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
counts = [0] * 8

for a in angles:
    # Convert angle to direction (0=N, 45=NE, 90=E, etc.)
    bin_idx = int((a + 22.5) // 45) % 8
    counts[bin_idx] += 1

for i, (b, c) in enumerate(zip(bins, counts)):
    bar = '#' * (c // 5)
    print(f"{b:>3}: {c:>3} {bar}")

# Select 5 sample images
sample_images = df['image'].unique()[:5]
print(f"\n=== Visualizing {len(sample_images)} sample images ===")

# Count max tubes in any sample image
max_tubes = max(len(df[df['image'] == img]) for img in sample_images)

# Create visualization
fig, axes = plt.subplots(len(sample_images), max_tubes, figsize=(5 * max_tubes, 4 * len(sample_images)))
if len(sample_images) == 1:
    axes = axes.reshape(1, -1)
if max_tubes == 1:
    axes = axes.reshape(-1, 1)

for row_idx, img_name in enumerate(sample_images):
    img_path = f'./images/{img_name}'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Cannot load {img_path}")
        continue

    img_tubes = df[df['image'] == img_name]

    for col_idx, (_, row) in enumerate(img_tubes.iterrows()):
        cx, cy = int(row['center_x']), int(row['center_y'])
        angle_deg = row['angle_deg']
        bbox_rot = row['bbox_rotation']

        # Crop 96x96 patch
        crop_size = 96
        half = crop_size // 2
        h, w = img.shape[:2]

        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        crop_x1 = max(0, int(x1))
        crop_y1 = max(0, int(y1))
        crop_x2 = min(w, int(x2))
        crop_y2 = min(h, int(y2))

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=0)

        # Draw arrows
        # Center in crop coordinates
        crop_cx = half
        crop_cy = half

        # Green arrow: angle_deg (GT)
        rad = np.radians(angle_deg)
        dx = 40 * np.cos(rad)
        dy = 40 * np.sin(rad)
        end_x = int(crop_cx + dx)
        end_y = int(crop_cy - dy)
        cv2.arrowedLine(crop, (crop_cx, crop_cy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

        # Blue arrow: bbox_rotation
        rad = np.radians(bbox_rot)
        dx = 40 * np.cos(rad)
        dy = 40 * np.sin(rad)
        end_x = int(crop_cx + dx)
        end_y = int(crop_cy - dy)
        cv2.arrowedLine(crop, (crop_cx, crop_cy), (end_x, end_y), (255, 0, 0), 3, tipLength=0.3)

        # Convert to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        axes[row_idx, col_idx].imshow(crop_rgb)
        axes[row_idx, col_idx].axis('off')
        axes[row_idx, col_idx].set_title(f"GT:{angle_deg:.0f}° BBox:{bbox_rot:.0f}°", fontsize=10)

plt.tight_layout()
plt.savefig('angle_diagnosis.png', dpi=150)
print("Saved angle_diagnosis.png")