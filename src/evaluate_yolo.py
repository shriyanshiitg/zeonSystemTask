import cv2
import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Load model
model_path = './runs/obb/runs/tube_obb/weights/best.pt'
print(f"Loading model: {model_path}")

from ultralytics import YOLO
model = YOLO(model_path)

# Load GT
gt_df = pd.read_csv('annotations.csv')
image_files = sorted([f for f in os.listdir('./images') if f.endswith('.png')])
print(f"Evaluating on {len(image_files)} images")

conf_threshold = 0.25
distance_threshold = 20

def circular_distance(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def match_predictions(predictions, gt_boxes, distance_threshold=20):
    if len(predictions) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(predictions))), []

    # Build cost matrix
    cost_matrix = np.zeros((len(predictions), len(gt_boxes)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(gt_boxes):
            dist = np.sqrt((pred['cx'] - gt['cx'])**2 + (pred['cy'] - gt['cy'])**2)
            cost_matrix[i, j] = dist

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= distance_threshold:
            matches.append((i, j))
            matched_pred_indices.add(i)
            matched_gt_indices.add(j)

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred_indices]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gt_indices]

    return matches, unmatched_preds, unmatched_gts

# Evaluate
results = []
all_angle_errors = []

for img_name in image_files:
    img_path = f'./images/{img_name}'
    img = cv2.imread(img_path)

    img_gt = gt_df[gt_df['image'] == img_name].reset_index(drop=True)
    gt_boxes = [{'cx': row['center_x'], 'cy': row['center_y'], 'angle': row['angle_deg']} for _, row in img_gt.iterrows()]

    preds = model.predict(img_path, conf=conf_threshold, verbose=False)[0]

    predictions = []
    if preds.obb is not None:
        for box in preds.obb:
            xywhr = box.xywhr[0].cpu().numpy()
            cx = xywhr[0]
            cy = xywhr[1]
            angle_rad = xywhr[4]
            angle = (angle_rad * 180 / np.pi) % 360
            predictions.append({'cx': cx, 'cy': cy, 'angle': angle, 'conf': box.conf[0].cpu().item()})

    matches, fps, fns = match_predictions(predictions, gt_boxes, distance_threshold)

    for pred_idx, gt_idx in matches:
        err = circular_distance(predictions[pred_idx]['angle'], gt_boxes[gt_idx]['angle'])
        all_angle_errors.append(err)

    results.append({
        'image': img_name,
        'num_gt': len(gt_boxes),
        'num_preds': len(predictions),
        'tp': len(matches),
        'fp': len(fps),
        'fn': len(fns)
    })

# Overall metrics
total_tp = sum(r['tp'] for r in results)
total_fp = sum(r['fp'] for r in results)
total_fn = sum(r['fn'] for r in results)

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

mean_angle_error = np.mean(all_angle_errors) if all_angle_errors else 0
median_angle_error = np.median(all_angle_errors) if all_angle_errors else 0
within_10 = sum(1 for e in all_angle_errors if e <= 10) / len(all_angle_errors) * 100 if all_angle_errors else 0
within_20 = sum(1 for e in all_angle_errors if e <= 20) / len(all_angle_errors) * 100 if all_angle_errors else 0
within_30 = sum(1 for e in all_angle_errors if e <= 30) / len(all_angle_errors) * 100 if all_angle_errors else 0

# Save CSV
results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)
print("Saved evaluation_results.csv")

# Save summary
with open('evaluation_summary.txt', 'w') as f:
    f.write("=== YOLO OBB Evaluation Results ===\n\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Confidence threshold: {conf_threshold}\n")
    f.write(f"Distance threshold: {distance_threshold}px\n\n")
    f.write(f"Total images: {len(image_files)}\n")
    f.write(f"Total GT tubes: {len(gt_df)}\n")
    f.write(f"Total predictions: {sum(r['num_preds'] for r in results)}\n")
    f.write(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}\n\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write(f"Mean angle error: {mean_angle_error:.2f} deg\n")
    f.write(f"Median angle error: {median_angle_error:.2f} deg\n")
    f.write(f"Within 10 deg: {within_10:.1f}%\n")
    f.write(f"Within 20 deg: {within_20:.1f}%\n")
    f.write(f"Within 30 deg: {within_30:.1f}%\n")
print("Saved evaluation_summary.txt")

# Visualization
sample_images = image_files[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, img_name in enumerate(sample_images):
    img_path = f'./images/{img_name}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gt = gt_df[gt_df['image'] == img_name]
    for _, row in img_gt.iterrows():
        cx, cy = int(row['center_x']), int(row['center_y'])
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), 2)
        angle = row['angle_deg']
        rad = np.radians(angle)
        dx = 30 * np.cos(rad)
        dy = 30 * np.sin(rad)
        end_x = int(cx + dx)
        end_y = int(cy - dy)
        cv2.line(img, (cx, cy), (end_x, end_y), (0, 255, 0), 2)

    preds = model.predict(img_path, conf=conf_threshold, verbose=False)[0]
    if preds.obb is not None:
        for box in preds.obb:
            xywhr = box.xywhr[0].cpu().numpy()
            cx, cy = int(xywhr[0]), int(xywhr[1])
            cv2.circle(img, (cx, cy), 8, (255, 0, 0), 2)
            angle = (xywhr[4] * 180 / np.pi) % 360
            rad = np.radians(angle)
            dx = 25 * np.cos(rad)
            dy = 25 * np.sin(rad)
            end_x = int(cx + dx)
            end_y = int(cy - dy)
            cv2.line(img, (cx, cy), (end_x, end_y), (255, 0, 0), 2)

    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(f"{img_name[:15]}\nGT: {len(img_gt)}, Pred: {len(preds.obb) if preds.obb else 0}", fontsize=10)

plt.tight_layout()
plt.savefig('evaluation_viz.png', dpi=150)
print("Saved evaluation_viz.png")

# Print summary
print(f"\n=== Evaluation Summary ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Angle Error: {mean_angle_error:.2f} deg")
print(f"Median Angle Error: {median_angle_error:.2f} deg")
print(f"Within 10°: {within_10:.1f}%")
print(f"Within 20°: {within_20:.1f}%")
print(f"Within 30°: {within_30:.1f}%")