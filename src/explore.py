import cv2
import numpy as np
import pandas as pd

# Read annotations
df = pd.read_csv('annotations.csv')

# Print statistics
print(f"Total images: {df['image'].nunique()}")
print(f"Total tubes: {len(df)}")
tubes_per_image = df.groupby('image').size()
print(f"Min tubes per image: {tubes_per_image.min()}")
print(f"Max tubes per image: {tubes_per_image.max()}")
print(f"Min angle_deg: {df['angle_deg'].min()}")
print(f"Max angle_deg: {df['angle_deg'].max()}")
print(f"Mean angle_deg: {df['angle_deg'].mean():.2f}")

# Load first sample image
first_image_name = df['image'].iloc[0]
image_path = f"./images/{first_image_name}"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Get tubes for first image
first_image_tubes = df[df['image'] == first_image_name]

# Draw each tube
for _, row in first_image_tubes.iterrows():
    cx, cy = int(row['center_x']), int(row['center_y'])
    angle = row['angle_deg']

    # Draw red circle at center
    cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

    # Draw green line showing angle direction (length 30px)
    angle_rad = np.radians(angle)
    dx = 30 * np.cos(angle_rad)
    dy = 30 * np.sin(angle_rad)
    end_x = int(cx + dx)
    end_y = int(cy - dy)  # OpenCV y-axis is inverted
    cv2.line(img, (cx, cy), (end_x, end_y), (0, 255, 0), 2)

# Save visualization
cv2.imwrite('explore_output.png', img)
print("Saved visualization to explore_output.png")