import pandas as pd
import numpy as np
import os
import shutil
import yaml

# Read annotations
df = pd.read_csv('annotations.csv')

# Get unique images, sorted alphabetically
unique_images = sorted(df['image'].unique())
print(f"Total unique images: {len(unique_images)}")

# Split: first 60 → train, remaining 10 → val
train_images = unique_images[:60]
val_images = unique_images[60:]
print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

# Create folder structure
os.makedirs('./yolo_dataset/images/train', exist_ok=True)
os.makedirs('./yolo_dataset/images/val', exist_ok=True)
os.makedirs('./yolo_dataset/labels/train', exist_ok=True)
os.makedirs('./yolo_dataset/labels/val', exist_ok=True)

# Image dimensions
IMG_W, IMG_H = 640, 480

def obb_to_polygon(cx, cy, w, h, angle_deg):
    """Convert OBB to 4 corner points (normalized)."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Half dimensions in pixel space
    hw = w / 2
    hh = h / 2

    # 4 corners relative to center (before rotation)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

    # Rotation matrix
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ R.T

    # Add center (in pixel space)
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    # Normalize
    rotated[:, 0] /= IMG_W
    rotated[:, 1] /= IMG_H

    return rotated.flatten()

# Copy images and create labels
def process_split(images, split):
    label_dir = f'./yolo_dataset/labels/{split}'
    img_dir = f'./yolo_dataset/images/{split}'

    for img_name in images:
        # Copy image
        src_img = f'./images/{img_name}'
        dst_img = f'{img_dir}/{img_name}'
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
            continue

        # Get annotations for this image
        img_annots = df[df['image'] == img_name]

        # Write label file in polygon format (8 columns)
        label_name = img_name.replace('.png', '.txt')
        label_path = f'{label_dir}/{label_name}'

        lines = []
        for _, row in img_annots.iterrows():
            cx, cy = row['center_x'], row['center_y']
            w, h = row['bbox_w'], row['bbox_h']
            # Use angle_deg column for rotation
            angle_deg = row['angle_deg']

            polygon = obb_to_polygon(cx, cy, w, h, angle_deg)
            # Class 0 = tube - format: class x1 y1 x2 y2 x3 y3 x4 y4 (8 columns)
            line = "0 " + " ".join([f"{v:.6f}" for v in polygon])
            lines.append(line)

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

    print(f"  {split}: {len(images)} images")

# Process train and val
process_split(train_images, 'train')
process_split(val_images, 'val')

# Count total annotations
train_annots = len(df[df['image'].isin(train_images)])
val_annots = len(df[df['image'].isin(val_images)])

# Write data.yaml
data_yaml = {
    'path': './yolo_dataset',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': ['tube']
}

with open('./yolo_dataset/data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# Print summary
print(f"\n=== Summary ===")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
print(f"Total annotations: {len(df)}")
print(f"Train annotations: {train_annots}")
print(f"Val annotations: {val_annots}")
print(f"\nDataset created at ./yolo_dataset/")