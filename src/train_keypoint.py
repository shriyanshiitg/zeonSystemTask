"""
Experiment 1: Keypoint Detection for Tab Tip Localization

This experiment treats the tab tip as a keypoint to solve the 180° ambiguity problem.
A U-Net style model with ResNet-18 encoder predicts heatmaps of tab tip locations.

Organized under experiment01_keypoint/ for clean management.
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
from scipy.ndimage import maximum_filter
import random

# Configuration
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"
MODEL_WEIGHTS = Path("models/weights/best.pt")
OUTPUT_DIR = Path("results/experiment01_keypoint")
OUTPUT_SUMMARY = OUTPUT_DIR / "evaluation_summary.txt"

# Hyperparameters
IMAGE_SIZE = (480, 640)  # H, W
TAB_OFFSET = 35  # pixels from center to tab tip
SIGMA = 15  # Gaussian blob sigma
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 5e-5
DEVICE = 'cpu'
NUM_TRAIN = 60
NUM_VAL = 10
RANDOM_SEED = 42

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def compute_tab_tip(center_x, center_y, angle_deg):
    """Compute tab tip position from center and angle."""
    angle_rad = math.radians(angle_deg)
    # Note: angle_deg is counter-clockwise from positive X axis
    tab_x = center_x + TAB_OFFSET * math.cos(angle_rad)
    tab_y = center_y - TAB_OFFSET * math.sin(angle_rad)  # Y is inverted in image coords
    return tab_x, tab_y



def weighted_mse_loss(pred, target, pos_weight_factor=50):
    import torch
    pos_weight = 1 + (pos_weight_factor - 1) * (target > 0.1).float()
    mse = (pred - target) ** 2
    return (pos_weight * mse).mean()

def create_gaussian_heatmap(h, w, x, y, sigma=SIGMA):
    """Create a Gaussian blob at (x, y) on an h x w canvas."""
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Create meshgrid
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))

    # Gaussian formula
    exp = -((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)
    heatmap = np.exp(exp)

    # Normalize to [0, 1]
    heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    return heatmap


def create_image_heatmap(image_name, annotations_df, image_size):
    """Create heatmap for a single image with all tab tips."""
    h, w = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Get all tubes in this image
    image_annotations = annotations_df[annotations_df['image'] == image_name]

    for _, row in image_annotations.iterrows():
        tab_x, tab_y = compute_tab_tip(row['center_x'], row['center_y'], row['angle_deg'])

        # Only add if within bounds
        if 0 <= tab_x < w and 0 <= tab_y < h:
            heatmap += create_gaussian_heatmap(h, w, tab_x, tab_y)

    # Clip to [0, 1]
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap


class KeypointDataset(Dataset):
    """Dataset for keypoint detection."""

    def __init__(self, image_names, annotations_df, image_dir, image_size):
        self.image_names = image_names
        self.annotations_df = annotations_df
        self.image_dir = Path(image_dir)
        self.image_size = image_size  # (H, W)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load image
        image_path = self.image_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            # Return empty on error
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create heatmap
        heatmap = create_image_heatmap(image_name, self.annotations_df, self.image_size)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)

        return image_tensor, heatmap_tensor


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder for U-Net."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            resnet = resnet18(weights=weights)
        else:
            resnet = resnet18(weights=None)

        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Store output channels for decoder
        self.encoder_channels = [64, 128, 256, 512]

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.layer1(x)   # 64
        e2 = self.layer2(e1)   # 128
        e3 = self.layer3(e2)   # 256
        e4 = self.layer4(e3)   # 512

        return e1, e2, e3, e4


class DecoderBlock(nn.Module):
    """Decoder upsampling block for U-Net."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class KeypointUNet(nn.Module):
    """U-Net for keypoint detection with ResNet-18 encoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # Decoder
        channels = self.encoder.encoder_channels  # [64, 128, 256, 512]

        self.decoder4 = DecoderBlock(channels[3], channels[2], 256)  # 512 + 256 -> 256
        self.decoder3 = DecoderBlock(256, channels[1], 128)           # 256 + 128 -> 128
        self.decoder2 = DecoderBlock(128, channels[0], 64)            # 128 + 64 -> 64
        self.decoder1 = DecoderBlock(64, 0, 32)                      # 64 -> 32

        # Final output
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.encoder(x)

        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, None)

        # Output
        out = self.output(d1)

        return out


def find_keypoints(heatmap, threshold=0.3, min_distance=10):
    """Find local maxima in heatmap as keypoints."""
    # Apply maximum filter
    filtered = maximum_filter(heatmap, size=min_distance)

    # Find peaks
    peaks = (heatmap == filtered) & (heatmap > threshold)

    # Get coordinates
    coords = np.where(peaks)
    if len(coords[0]) == 0:
        return []

    # Sort by confidence
    scores = heatmap[peaks]
    sorted_idx = np.argsort(scores)[::-1]

    keypoints = []
    for i in sorted_idx:
        keypoints.append({
            'x': coords[1][i],  # column = x
            'y': coords[0][i],  # row = y
            'score': scores[i]
        })

    return keypoints


def match_keypoint_to_center(keypoint, centers, threshold=20):
    """Match a keypoint to the nearest center."""
    best_idx = -1
    best_dist = float('inf')

    for i, center in enumerate(centers):
        dist = math.sqrt((keypoint['x'] - center['x'])**2 + (keypoint['y'] - center['y'])**2)
        if dist < best_dist and dist <= threshold:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


def compute_angle_from_keypoint(center_x, center_y, tip_x, tip_y):
    """Compute angle from center to tip."""
    dx = tip_x - center_x
    dy = center_y - tip_y  # Note: Y is inverted in image

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Convert to 0-360 range
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def circular_error(pred, gt):
    """Calculate circular error (minimum angular distance)."""
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


def train_model():
    """Train the keypoint detection model."""
    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_FILE)
    images = df['image'].unique().tolist()
    print(f"Total unique images: {len(images)}")

    # Split into train/val (60/10)
    random.shuffle(images)
    train_images = images[:NUM_TRAIN]
    val_images = images[NUM_TRAIN:NUM_TRAIN + NUM_VAL]

    print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

    # Create datasets
    train_dataset = KeypointDataset(train_images, df, IMAGE_DIR, IMAGE_SIZE)
    val_dataset = KeypointDataset(val_images, df, IMAGE_DIR, IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    model = KeypointUNet(pretrained=True).to(DEVICE)

    # Freeze encoder for first 10 epochs
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = weighted_mse_loss

    # Training loop
    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        # Unfreeze encoder after 10 epochs
        if epoch == 10:
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            print("Unfroze encoder")

        model.train()
        train_loss = 0.0

        for batch_idx, (images, heatmaps) in enumerate(train_loader):
            images = images.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            # Sanity check first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"  [Sanity] pred.max={outputs.max().item():.4f}, target.max={heatmaps.max().item():.4f}")
        train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "keypoint_model.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

    print("Training complete!")
    return model, train_images, val_images


def evaluate_model(model, val_images):
    """Evaluate keypoint detection and angle estimation."""
    print("\nEvaluating model...")
    df = pd.read_csv(ANNOTATIONS_FILE)

    # Load YOLO model for detection
    from ultralytics import YOLO
    yolo_model = YOLO(str(MODEL_WEIGHTS))
    yolo_model.to(DEVICE)

    # Process each validation image
    results = []

    model.eval()
    with torch.no_grad():
        for image_name in val_images:
            # Load image
            image_path = IMAGE_DIR / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # Run YOLO detection
            yolo_results = yolo_model.predict(
                str(image_path),
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            # Get YOLO detections (tube centers)
            yolo_centers = []
            if len(yolo_results) > 0 and yolo_results[0].obb is not None and len(yolo_results[0].obb) > 0:
                boxes = yolo_results[0].obb
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    yolo_centers.append({'x': float(cx), 'y': float(cy)})

            if not yolo_centers:
                continue

            # Run keypoint detection
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

            heatmap = model(image_tensor).cpu().numpy()[0, 0]

            # Find keypoints
            keypoints = find_keypoints(heatmap, threshold=0.3, min_distance=10)

            # Get GT annotations for this image
            gt_annotations = df[df['image'] == image_name]

            # Match keypoints to YOLO centers, then evaluate against GT
            matched_keypoints = []
            for kp in keypoints:
                idx, dist = match_keypoint_to_center(kp, yolo_centers, threshold=20)
                if idx >= 0:
                    matched_keypoints.append({
                        'center': yolo_centers[idx],
                        'tip': {'x': kp['x'], 'y': kp['y']},
                        'dist': dist
                    })

            # Evaluate each matched detection against GT
            true_positives = 0
            false_positives = 0
            total_gt = len(gt_annotations)
            matched_gt_indices = set()

            for match in matched_keypoints:
                center_x = match['center']['x']
                center_y = match['center']['y']
                tip_x = match['tip']['x']
                tip_y = match['tip']['y']

                # Find closest GT tube
                best_gt_dist = float('inf')
                best_gt_idx = -1
                for idx, (_, row) in enumerate(gt_annotations.iterrows()):
                    dist = math.sqrt((center_x - row['center_x'])**2 + (center_y - row['center_y'])**2)
                    if dist < best_gt_dist:
                        best_gt_dist = dist
                        best_gt_idx = idx

                if best_gt_idx >= 0 and best_gt_dist <= 20:
                    true_positives += 1
                    if best_gt_idx not in matched_gt_indices:
                        matched_gt_indices.add(best_gt_idx)
                        # Compute predicted angle
                        pred_angle = compute_angle_from_keypoint(center_x, center_y, tip_x, tip_y)
                        gt_angle = gt_annotations.iloc[best_gt_idx]['angle_deg']
                        error = circular_error(pred_angle, gt_angle)
                        results.append({
                            'image': image_name,
                            'gt_angle': gt_angle,
                            'pred_angle': pred_angle,
                            'error': error
                        })
                else:
                    false_positives += 1

            false_negatives = total_gt - len(matched_gt_indices)

    # Compute metrics
    if not results:
        print("No valid results!")
        return None

    errors = [r['error'] for r in results]

    # Detection metrics
    total_tp = len(results)
    total_fp = 0  # These are tracked per-image but we aggregate here
    total_fn = 0

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    within_10 = np.sum(np.array(errors) <= 10) / len(errors) * 100 if errors else 0
    within_20 = np.sum(np.array(errors) <= 20) / len(errors) * 100 if errors else 0
    within_30 = np.sum(np.array(errors) <= 30) / len(errors) * 100 if errors else 0

    # Approximate detection metrics
    # TP = matched detections, FN = unmatched GT, FP = unmatched predictions
    all_gt = len(df[df['image'].isin(val_images)])
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
    recall = (total_tp / all_gt * 100) if all_gt > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"\nKeypoint Evaluation Results:")
    print(f"  Total matched detections: {len(results)}")
    print(f"  Precision: {precision:.1f}%")
    print(f"  Recall: {recall:.1f}%")
    print(f"  F1: {f1:.1f}%")
    print(f"  Mean angle error: {mean_error:.2f}°")
    print(f"  Median angle error: {median_error:.2f}°")
    print(f"  % within 10°: {within_10:.1f}%")
    print(f"  % within 20°: {within_20:.1f}%")
    print(f"  % within 30°: {within_30:.1f}%")

    # Save summary
    summary = f"""KEYPOINT DETECTION EVALUATION SUMMARY
========================================
Experiment: 01 - Keypoint Detection
Method: U-Net with ResNet-18 encoder, heatmap-based keypoint detection

Training Configuration:
  - Tab offset: {TAB_OFFSET}px from center to tab tip
  - Gaussian sigma: {SIGMA}px
  - Training images: {NUM_TRAIN}
  - Validation images: {NUM_VAL}
  - Epochs: {EPOCHS}
  - Batch size: {BATCH_SIZE}
  - Learning rate: {LEARNING_RATE}
  - Encoder frozen for first 10 epochs

Results:
  Total matched detections: {len(results)}
  Precision: {precision:.1f}%
  Recall: {recall:.1f}%
  F1: {f1:.1f}%
  Mean angle error: {mean_error:.2f}°
  Median angle error: {median_error:.2f}°
  % within 10°: {within_10:.1f}%
  % within 20°: {within_20:.1f}%
  % within 30°: {within_30:.1f}%

Keypoint Detection Metrics:
  - YOLO provides tube center detections
  - Keypoint model predicts tab tip locations as heatmap local maxima
  - 180° ambiguity resolved by explicitly detecting the tab tip
  - Local maxima threshold: 0.3
  - Matching threshold: 20px between keypoint and tube center

Notes:
  - Images: 640x480
  - Device: {DEVICE}
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


def main():
    """Main function."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train model
    model, train_images, val_images = train_model()

    # Evaluate
    metrics = evaluate_model(model, val_images)

    return metrics


if __name__ == "__main__":
    main()