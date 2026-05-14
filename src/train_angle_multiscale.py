"""
Experiment 3: Multi-Scale Angle Head Training

Tests whether larger crops improve angle prediction by training
separate ResNet-18 angle heads at 64x64, 96x96, and 128x128 resolutions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
import math

# Configuration
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"
MODEL_DIR = Path("models/weights")
OUTPUT_DIR = Path("results")
OUTPUT_FILE = OUTPUT_DIR / "multiscale_comparison.txt"

DEVICE = 'cpu'
CROP_SIZES = [64, 96, 128]
EPOCHS = 60
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
RANDOM_SEED = 42

# Set seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def circular_error(pred, gt):
    """Calculate circular error (minimum angular distance)."""
    diff = abs(pred - gt)
    return min(diff, 360 - diff)


class TubeAngleDataset(Dataset):
    """Dataset for tube angle regression with configurable crop size."""

    def __init__(self, df, images_dir, crop_size, augment=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.crop_size = crop_size
        self.augment = augment
        self.half = crop_size // 2

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def extract_crop(self, img, cx, cy):
        """Extract centered crop with padding."""
        h, w = img.shape[:2]
        x1 = int(cx) - self.half
        y1 = int(cy) - self.half
        x2 = int(cx) + self.half
        y2 = int(cy) + self.half

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w, x2)
        crop_y2 = min(h, y2)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)
        return crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row['image']
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load {img_path}")

        cx = row['center_x']
        cy = row['center_y']
        angle_deg = row['angle_deg']

        crop = self.extract_crop(img, cx, cy)

        if self.augment:
            # Random rotation (0-360)
            if np.random.random() < 0.5:
                rot_angle = np.random.uniform(-180, 180)
                M = cv2.getRotationMatrix2D((self.half, self.half), rot_angle, 1.0)
                crop = cv2.warpAffine(crop, M, (self.crop_size, self.crop_size),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                angle_deg = (angle_deg + rot_angle) % 360

            # Horizontal flip
            if np.random.random() < 0.5:
                crop = cv2.flip(crop, 1)
                angle_deg = (180 - angle_deg) % 360

            # Vertical flip
            if np.random.random() < 0.5:
                crop = cv2.flip(crop, 0)
                angle_deg = (360 - angle_deg) % 360

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_tensor = self.transform(crop_rgb)

        angle_rad = math.radians(angle_deg)
        sin_val = math.sin(angle_rad)
        cos_val = math.cos(angle_rad)

        return crop_tensor, torch.tensor([sin_val, cos_val], dtype=torch.float32)


class AngleHead(nn.Module):
    """ResNet-18 based angle regression head."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)


def train_model(train_df, val_df, crop_size, model_name):
    """Train a single angle head model."""
    print(f"\n{'='*60}")
    print(f"Training with crop_size={crop_size}x{crop_size}")
    print(f"{'='*60}")

    train_dataset = TubeAngleDataset(train_df, IMAGE_DIR, crop_size, augment=True)
    val_dataset = TubeAngleDataset(val_df, IMAGE_DIR, crop_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AngleHead().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_error = float('inf')
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for crops, targets in train_loader:
            crops = crops.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs[:, 0], targets[:, 0]) + criterion(outputs[:, 1], targets[:, 1])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_errors = []
        with torch.no_grad():
            for crops, targets in val_loader:
                crops = crops.to(DEVICE)
                outputs = model(crops)

                pred_sin = outputs[:, 0].cpu().numpy()
                pred_cos = outputs[:, 1].cpu().numpy()
                pred_angle = (np.arctan2(pred_sin, pred_cos) * 180 / np.pi) % 360

                gt_sin = targets[:, 0].numpy()
                gt_cos = targets[:, 1].numpy()
                gt_angle = (np.arctan2(gt_sin, gt_cos) * 180 / np.pi) % 360

                for p, g in zip(pred_angle, gt_angle):
                    val_errors.append(circular_error(p, g))

        mean_error = np.mean(val_errors) if val_errors else float('inf')

        if mean_error < best_val_error:
            best_val_error = mean_error
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}, Val Error: {mean_error:.2f}°")

    print(f"Best val error: {best_val_error:.2f}°")

    # Save best model
    torch.save(best_model_state, MODEL_DIR / f"{model_name}.pth")
    print(f"Saved to {MODEL_DIR / f'{model_name}.pth'}")

    return best_val_error


def evaluate_model(model, val_df, crop_size):
    """Evaluate model and return detailed metrics."""
    val_dataset = TubeAngleDataset(val_df, IMAGE_DIR, crop_size, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    errors = []

    with torch.no_grad():
        for crops, targets in val_loader:
            crops = crops.to(DEVICE)
            outputs = model(crops)

            pred_sin = outputs[0, 0].item()
            pred_cos = outputs[0, 1].item()
            pred_angle = math.degrees(math.atan2(pred_sin, pred_cos)) % 360

            gt_sin = targets[0, 0].item()
            gt_cos = targets[0, 1].item()
            gt_angle = math.degrees(math.atan2(gt_sin, gt_cos)) % 360

            error = circular_error(pred_angle, gt_angle)
            errors.append(error)

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    within_30 = np.sum(np.array(errors) <= 30) / len(errors) * 100

    return mean_error, median_error, within_30


def main():
    """Main function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations...")
    df = pd.read_csv(ANNOTATIONS_FILE)
    print(f"Total tubes: {len(df)}")

    # Train/val split (80/20) - IMAGE LEVEL, not tube level
    all_images = sorted(df['image'].unique())
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    train_images = all_images[:int(0.8 * len(all_images))]
    val_images = all_images[int(0.8 * len(all_images)):]

    train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['image'].isin(val_images)].reset_index(drop=True)

    # Verify no leakage
    assert len(set(train_images) & set(val_images)) == 0, "LEAKAGE!"

    print(f"Train: {len(train_images)} images, {len(train_df)} tubes")
    print(f"Val: {len(val_images)} images, {len(val_df)} tubes")

    # Train models at each scale
    results = []
    best_error = float('inf')
    best_model_name = None

    for crop_size in CROP_SIZES:
        model_name = f"angle_head_{crop_size}"
        val_error = train_model(train_df, val_df, crop_size, model_name)

        # Load best model and evaluate
        model = AngleHead().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}.pth", map_location=DEVICE))

        mean_err, median_err, within_30 = evaluate_model(model, val_df, crop_size)

        results.append({
            'crop_size': crop_size,
            'mean_error': mean_err,
            'median_error': median_err,
            'within_30': within_30
        })

        if mean_err < best_error:
            best_error = mean_err
            best_model_name = model_name

    # Find best model
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")

    # Comparison table
    print(f"\n{'Crop Size':<12} {'Mean Error':<12} {'Median Error':<14} {'Within 30°':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['crop_size']:<12} {r['mean_error']:<12.2f}° {r['median_error']:<14.2f}° {r['within_30']:<10.1f}%")

    # Save best model
    import shutil
    shutil.copy(MODEL_DIR / f"{best_model_name}.pth", MODEL_DIR / "angle_head_best.pth")
    print(f"\nBest model: {best_model_name}.pth (copied to angle_head_best.pth)")

    # Save results
    summary = f"""MULTI-SCALE ANGLE HEAD COMPARISON
===================================
Experiment: 03 - Larger Crops + Higher Resolution

Training Configuration:
  - Epochs: {EPOCHS}
  - Batch size: {BATCH_SIZE}
  - Learning rate: {LEARNING_RATE}
  - Optimizer: Adam
  - Augmentation: Random rotation, H/V flip, color jitter (brightness=0.3, contrast=0.3)

Data Split:
  - Train: {len(train_df)} tubes
  - Val: {len(val_df)} tubes

RESULTS COMPARISON
------------------
{'Crop Size':<12} {'Mean Error':<12} {'Median Error':<14} {'Within 30°':<10}
{'-'*50}
"""
    for r in results:
        summary += f"{r['crop_size']:<12} {r['mean_error']:<12.2f}° {r['median_error']:<14.2f}° {r['within_30']:<10.1f}%\n"

    summary += f"""
Best Model: {best_model_name}.pth
Best Mean Error: {best_error:.2f}°
Saved to: models/weights/angle_head_best.pth

Notes:
  - Larger crops may capture more tab detail
  - Trade-off: larger crops are slower to process and may include more background noise
"""

    with open(OUTPUT_FILE, 'w') as f:
        f.write(summary)

    print(f"\nSaved results to {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    main()