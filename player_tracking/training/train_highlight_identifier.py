import pandas as pd
import torch
import torchvision
import torch.nn as nn
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import timm
from tqdm import tqdm
import torchvision.transforms as transforms
import json
import numpy as np

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

class PlayerTrackingDataset(Dataset):
    def __init__(self, df, root="data", transform=None, sample_frames=10, frames_dir="extracted_frames"):
        """
        Dataset that loads pre-extracted frames instead of opening videos.
        
        Args:
            df: DataFrame with columns: video_path, start, end, track, and 'original_idx' 
                (original CSV row index used when extracting frames)
            root: Root directory for videos (not used when loading from frames)
            transform: Image transforms
            sample_frames: Number of frames per row (must match extraction)
            frames_dir: Directory containing extracted frames
        """
        self.df = df.copy()
        # Ensure original_idx column exists (from original CSV row index)
        if 'original_idx' not in self.df.columns:
            raise ValueError("DataFrame must have 'original_idx' column. Make sure to add it after reading CSV.")
        self.root = root
        self.transform = transform
        self.sample_frames = sample_frames
        self.frames_dir = frames_dir
    
    def __len__(self):
        return self.df.shape[0] * self.sample_frames
    
    def __getitem__(self, idx):
        row_idx = idx // self.sample_frames
        frame_idx = idx % self.sample_frames
        
        row = self.df.iloc[row_idx]
        original_row_idx = int(row["original_idx"])
        
        # Construct frame filename: {original_row_idx}_{frame_idx}.jpg
        frame_filename = f"{original_row_idx}_{frame_idx}.jpg"
        frame_path = os.path.join(self.frames_dir, frame_filename)
        
        # Load image
        try:
            image = Image.open(frame_path).convert('RGB')
        except Exception as e:
            # Fallback: try to load any frame from this row
            fallback_frame_idx = random.randint(0, self.sample_frames - 1)
            fallback_filename = f"{original_row_idx}_{fallback_frame_idx}.jpg"
            fallback_path = os.path.join(self.frames_dir, fallback_filename)
            image = Image.open(fallback_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor([row["track"]], dtype=torch.float32)

df = pd.read_csv("track_frame.csv")
# Preserve original CSV row index for loading extracted frames
df['original_idx'] = df.index
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PlayerTrackingDataset(df_train, root="data", transform=transform, frames_dir="extracted_frames")
test_dataset = PlayerTrackingDataset(df_test, root="data", transform=transform_test, frames_dir="extracted_frames")

model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)
model.train()  # Set to training mode

batch_size = 64
lr = 1e-3
epochs = 15
patience = 3
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss().to(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []
train_auroc = []
val_auroc = []
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()  # Ensure model is in training mode
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch+1}/{epochs}")
    total_loss = 0
    weight = 0
    total_correct = 0
    all_train_outputs = []
    all_train_labels = []
    
    for i, (images, labels) in pbar:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.shape[0]
        weight += batch_size
        total_loss += loss.item() * batch_size  # Multiply by current batch size, not cumulative weight

        # Fix accuracy calculation: compare predictions with actual labels
        predictions = (outputs > 0.5).float()
        total_correct += (predictions == labels).sum().item()

        # Store outputs and labels for AUROC calculation
        all_train_outputs.append(outputs.detach().cpu().numpy())
        all_train_labels.append(labels.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

        # Compute AUROC incrementally for progress bar
        if len(all_train_outputs) > 0:
            current_outputs = np.concatenate(all_train_outputs, axis=0).flatten()
            current_labels = np.concatenate(all_train_labels, axis=0).flatten()
            current_auroc = roc_auc_score(current_labels, current_outputs)
            pbar.set_description(f"Train Epoch {epoch+1}/{epochs} Loss: {total_loss / weight:.4f} Acc: {total_correct / weight:.4f} AUROC: {current_auroc:.4f}")
        else:
            pbar.set_description(f"Train Epoch {epoch+1}/{epochs} Loss: {total_loss / weight:.4f} Acc: {total_correct / weight:.4f}")
        pbar.update(1)

    # Calculate final AUROC for training
    all_train_outputs = np.concatenate(all_train_outputs, axis=0).flatten()
    all_train_labels = np.concatenate(all_train_labels, axis=0).flatten()
    train_auroc_score = roc_auc_score(all_train_labels, all_train_outputs)
    
    train_loss.append(total_loss / weight)
    train_accuracy.append(total_correct / weight)
    train_auroc.append(train_auroc_score)
    # validate
    model.eval()  # Set to evaluation mode for validation
    with torch.no_grad():
        total_loss = 0
        weight = 0
        total_correct = 0
        all_val_outputs = []
        all_val_labels = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Val Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.shape[0]
            weight += batch_size
            total_loss += loss.item() * batch_size  # Multiply by current batch size, not cumulative weight

            # Fix accuracy calculation: compare predictions with actual labels
            predictions = (outputs > 0.5).float()
            total_correct += (predictions == labels).sum().item()

            # Store outputs and labels for AUROC calculation
            all_val_outputs.append(outputs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

            # Compute AUROC incrementally for progress bar
            if len(all_val_outputs) > 0:
                current_outputs = np.concatenate(all_val_outputs, axis=0).flatten()
                current_labels = np.concatenate(all_val_labels, axis=0).flatten()
                current_auroc = roc_auc_score(current_labels, current_outputs)
                pbar.set_description(f"Val Epoch {epoch+1}/{epochs} Loss: {total_loss / weight:.4f} Acc: {total_correct / weight:.4f} AUROC: {current_auroc:.4f}")
            else:
                pbar.set_description(f"Val Epoch {epoch+1}/{epochs} Loss: {total_loss / weight:.4f} Acc: {total_correct / weight:.4f}")
            pbar.update(1)

        # Calculate final AUROC for validation
        all_val_outputs = np.concatenate(all_val_outputs, axis=0).flatten()
        all_val_labels = np.concatenate(all_val_labels, axis=0).flatten()
        val_auroc_score = roc_auc_score(all_val_labels, all_val_outputs)

        val_loss.append(total_loss / weight)
        val_accuracy.append(total_correct / weight)
        val_auroc.append(val_auroc_score)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_accuracy[-1]:.4f}, Train AUROC: {train_auroc[-1]:.4f}")
        print(f"Epoch {epoch+1} - Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_accuracy[-1]:.4f}, Val AUROC: {val_auroc[-1]:.4f}")

    
    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience -= 1
    
    if patience == 0:
        break
    
    torch.save(model.state_dict(), f"last_model.pth")

    with open("train_metrics.json", "w+") as f:
        json.dump({
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            "train_auroc": train_auroc,
            "val_loss": val_loss, 
            "val_accuracy": val_accuracy,
            "val_auroc": val_auroc
        }, f, indent=2)
    


