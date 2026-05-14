import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from ultralytics import YOLO

device = 'cpu'

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO('./runs/obb/runs/tube_obb/weights/best.pt')

# Load angle head
print("Loading angle head...")
class AngleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)

angle_model = AngleHead().to(device)
angle_model.load_state_dict(torch.load('angle_head.pth', map_location=device))
angle_model.eval()

# Transform for angle head
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load GT
gt_df = pd.read_csv('annotations.csv')
image_files = sorted([f for f in os.listdir('./images') if f.endswith('.png')])
print(f"Evaluating on {len(image_files)} images")

conf_threshold = 0.25
distance_threshold = 20

def circular_distance(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def crop_and_predict_angle(img, cx, cy, model):
    """Crop 64x64 around center and predict angle."""
    crop_size = 64
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

    # Convert to RGB and predict
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(crop)
        sin_pred = outputs[0, 0].item()
        cos_pred = outputs[0, 1].item()
        angle = (np.arctan2(sin_pred, cos_pred) * 180 / np.pi) % 360

    return angle

def match_predictions(predictions, gt_boxes, distance_threshold=20):
    if len(predictions) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(predictions))), []

    cost_matrix = np.zeros((len(predictions), len(gt_boxes)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(gt_boxes):
            dist = np.sqrt((pred['cx'] - gt['cx'])**2 + (pred['cy'] - gt['cy'])**2)
            cost_matrix[i, j] = dist

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

    # YOLO detection
    preds = yolo_model.predict(img_path, conf=conf_threshold, verbose=False)[0]

    predictions = []
    if preds.obb is not None:
        for box in preds.obb:
            xywhr = box.xywhr[0].cpu().numpy()
            cx = xywhr[0]
            cy = xywhr[1]

            # Crop and predict angle with angle head
            angle = crop_and_predict_angle(img, int(cx), int(cy), angle_model)

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

# Save summary
with open('combined_evaluation_summary.txt', 'w') as f:
    f.write("=== Combined YOLO + Angle Head Evaluation Results ===\n\n")
    f.write(f"YOLO model: ./runs/obb/runs/tube_obb/weights/best.pt\n")
    f.write(f"Angle head: angle_head.pth\n")
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

print("\n=== Combined Evaluation Summary ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Angle Error: {mean_angle_error:.2f} deg")
print(f"Median Angle Error: {median_angle_error:.2f} deg")
print(f"Within 10°: {within_10:.1f}%")
print(f"Within 20°: {within_20:.1f}%")
print(f"Within 30°: {within_30:.1f}%")
print("\nSaved combined_evaluation_summary.txt")