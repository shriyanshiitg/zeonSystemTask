import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

device = 'cpu'

# Load annotations
df = pd.read_csv('annotations.csv')
print(f"Total tubes: {len(df)}")

# Dataset class
class TubeAngleDataset(Dataset):
    def __init__(self, df, images_dir='./images', augment=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.augment = augment

        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(180),  # Full rotation
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load {img_path}")

        cx, cy = int(row['center_x']), int(row['center_y'])
        angle_deg = row['angle_deg']

        # Crop 64x64 patch with padding
        crop_size = 64
        half = crop_size // 2
        h, w = img.shape[:2]

        # Calculate crop bounds
        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        # Pad if needed
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        # Extract crop
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w, x2)
        crop_y2 = min(h, y2)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Pad
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=0)

        # Apply augmentation and update angle
        if self.augment:
            # Random rotation
            if np.random.random() < 0.5:
                angle_rad = np.radians(angle_deg)
                # Get rotation angle from transform
                rot_angle = np.random.uniform(-180, 180)
                new_angle_deg = (angle_deg + rot_angle) % 360

                # Rotate crop manually
                M = cv2.getRotationMatrix2D((32, 32), rot_angle, 1.0)
                crop = cv2.warpAffine(crop, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                angle_deg = new_angle_deg

            # Random flips
            if np.random.random() < 0.5:
                crop = cv2.flip(crop, 1)  # horizontal
                angle_deg = (180 - angle_deg) % 360
            if np.random.random() < 0.5:
                crop = cv2.flip(crop, 0)  # vertical
                angle_deg = (-angle_deg) % 360

        # Convert to RGB and apply transforms
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = self.transform(crop)

        # Convert angle to sin/cos
        angle_rad = np.radians(angle_deg)
        sin_val = np.sin(angle_rad)
        cos_val = np.cos(angle_rad)

        return crop, torch.tensor([sin_val, cos_val], dtype=torch.float32)

# Build dataset
full_dataset = TubeAngleDataset(df, augment=False)

# Train/val split
torch.manual_seed(42)
indices = torch.randperm(len(full_dataset))
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices.tolist())
val_dataset = torch.utils.data.Subset(full_dataset, val_indices.tolist())

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Create augmented dataset for training
train_dataset_aug = TubeAngleDataset(df.iloc[train_indices.tolist()], augment=True)

train_loader = DataLoader(train_dataset_aug, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
class AngleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)

model = AngleHead().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
print("\nTraining...")
train_losses = []
val_errors = []

for epoch in range(50):
    # Train
    model.train()
    epoch_loss = 0
    for crops, targets in train_loader:
        crops = crops.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(crops)
        loss = criterion(outputs[:, 0], targets[:, 0]) + criterion(outputs[:, 1], targets[:, 1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validate
    model.eval()
    angle_errors = []
    with torch.no_grad():
        for crops, targets in val_loader:
            crops = crops.to(device)
            outputs = model(crops)

            # Convert predictions to angle
            pred_sin = outputs[:, 0].cpu().numpy()
            pred_cos = outputs[:, 1].cpu().numpy()
            pred_angle = (np.arctan2(pred_sin, pred_cos) * 180 / np.pi) % 360

            # Convert targets to angle
            gt_sin = targets[:, 0].numpy()
            gt_cos = targets[:, 1].numpy()
            gt_angle = (np.arctan2(gt_sin, gt_cos) * 180 / np.pi) % 360

            # Circular error
            for p, g in zip(pred_angle, gt_angle):
                err = min(abs(p - g), 360 - abs(p - g))
                angle_errors.append(err)

    mean_err = np.mean(angle_errors) if angle_errors else 0
    val_errors.append(mean_err)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.6f}, Val Angle Error: {mean_err:.2f}°")

# Final metrics
print("\n=== Training Complete ===")
print(f"Train loss (last 5 epochs): {train_losses[-5:]}")
print(f"Val mean angle error: {val_errors[-1]:.2f}°")

# Save model
torch.save(model.state_dict(), 'angle_head.pth')
print("Saved model to angle_head.pth")