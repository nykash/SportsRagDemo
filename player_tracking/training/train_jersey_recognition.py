#!/usr/bin/env python3
"""
Train a jersey number classifier using EfficientNet
Extracts cropped jersey regions from quadrilateral annotations and trains a classification model.
Predicts first and second digits separately (20 outputs total).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm
import timm
from PIL import Image
import random
from sklearn.metrics import roc_auc_score


class JerseyNumberDataset(Dataset):
    """Dataset for jersey number classification from combined dataset folder"""
    
    def __init__(
        self,
        dataset_dir: Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True
    ):
        """
        Args:
            dataset_dir: Path to combined dataset directory (with train/val/test subdirs)
            split: "train", "val", or "test"
            image_size: Target image size for model input
            augment: Whether to apply data augmentation
        """
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.dataset_dir = Path(dataset_dir)
        
        # Load images from the split directory
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Find all images
        image_files = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
        
        # Extract number from filename
        # Format: "quad_xxx_num14.jpg" or "jersey_numbers_xxx_num24.jpg" or just "xxx_num14.jpg"
        self.samples = []  # List of (image_path, number)
        
        for img_path in image_files:
            # Try to extract number from filename
            # Look for pattern: _numXX or numXX
            filename = img_path.stem
            number = None
            
            # Try pattern: _numXX
            if '_num' in filename:
                try:
                    num_str = filename.split('_num')[-1]
                    number = int(num_str)
                except (ValueError, IndexError):
                    pass
            
            # If not found, try to parse from end of filename
            if number is None:
                # Try to find number at the end
                parts = filename.split('_')
                for part in reversed(parts):
                    try:
                        number = int(part)
                        if 0 <= number <= 99:
                            break
                    except ValueError:
                        continue
            
            if number is not None and 0 <= number <= 99:
                self.samples.append((img_path, number))
            else:
                print(f"Warning: Could not extract number from {img_path.name}, skipping")
        
        # Create transforms
        self._create_transforms()
        
        print(f"Loaded {len(self.samples)} samples from {split_dir}")
    
    
    def _create_transforms(self):
        """Create image transforms"""
        # Basic normalization for CLIP models
        self.normalize = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return image
        
        h, w = image.shape[:2]
        
        # Random grayscale (convert to grayscale and back to RGB)
        if random.random() < 0.3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Convert back to RGB by stacking the grayscale channel
            image = np.stack([gray, gray, gray], axis=-1)
        
        
        # Random brightness/contrast (more aggressive)
        if random.random() < 0.7:
            alpha = random.uniform(0.6, 1.4)  # contrast (more range)
            beta = random.uniform(-40, 40)  # brightness (more range)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random saturation adjustment (convert to HSV, adjust, convert back)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Adjust saturation
            saturation_factor = random.uniform(0.5, 1.5)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random hue shift
        if random.random() < 0.3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = random.uniform(-10, 10)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 179  # OpenCV HSV H range is [0, 179]
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random shear
        if random.random() < 0.5:
            # Shear transformation
            shear_x = random.uniform(-0.2, 0.2)
            shear_y = random.uniform(-0.2, 0.2)
            
            # Create shear matrix
            M = np.array([
                [1, shear_x, -shear_x * w / 2],
                [shear_y, 1, -shear_y * h / 2]
            ], dtype=np.float32)
            
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random blur (motion blur or Gaussian blur)
        if random.random() < 0.4:
            blur_type = random.choice(['gaussian', 'motion'])
            
            if blur_type == 'gaussian':
                # Gaussian blur - simulates out-of-focus
                kernel_size = random.choice([3, 5, 7])  # Must be odd
                sigma = random.uniform(0.5, 2.0)
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            else:
                # Motion blur - simulates camera shake or fast movement
                kernel_size = random.choice([5, 7, 9, 11])
                angle = random.uniform(0, 180)
                image = self._apply_motion_blur(image, kernel_size, angle)
        
        # Random fold simulation (simulating fabric folds/creases)
        if random.random() < 0.3:
            image = self._simulate_fold(image)
        
        # Ensure image is still in valid range [0, 255]
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _apply_motion_blur(self, image, kernel_size, angle):
        """Apply motion blur to simulate camera shake or fast movement."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Angle in radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate kernel center
        center = kernel_size // 2
        
        # Draw a line in the kernel
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        for i in range(kernel_size):
            x = int(center + (i - center) * cos_angle)
            y = int(center + (i - center) * sin_angle)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        
        # Normalize kernel
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            # Fallback to simple horizontal blur
            kernel[center, :] = 1.0 / kernel_size
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def _simulate_fold(self, image):
        """Simulate a fold/crease in the jersey by removing a line and sliding pieces together.
        
        This simulates fabric folding where part of the jersey is obscured or distorted.
        Steps:
        1. Choose a random line (horizontal or vertical)
        2. Remove a strip at that line
        3. Slide the two pieces together
        4. Blend at the seam for smooth transition
        """
        h, w = image.shape[:2]
        
        # Choose fold orientation (horizontal or vertical)
        is_horizontal = random.random() < 0.5
        
        if is_horizontal:
            # Horizontal fold - split top and bottom
            fold_line = random.randint(int(h * 0.3), int(h * 0.7))
            fold_width = random.randint(2, max(4, int(h * 0.1)))  # Width of removed strip
            
            # Split into top and bottom parts
            top_part = image[:fold_line, :].copy()
            bottom_part = image[fold_line + fold_width:, :].copy()
            
            # Create new image by sliding pieces together
            new_image = np.zeros_like(image)
            
            # Place top part (unchanged)
            new_image[:fold_line, :] = top_part
            
            # Place bottom part shifted up (removing the fold strip)
            remaining_height = h - fold_line
            bottom_height = min(bottom_part.shape[0], remaining_height)
            if bottom_height > 0:
                new_image[fold_line:fold_line + bottom_height, :] = bottom_part[:bottom_height, :]
        
        else:
            # Vertical fold - split left and right
            fold_line = random.randint(int(w * 0.3), int(w * 0.7))
            fold_width = random.randint(2, max(4, int(w * 0.1)))  # Width of removed strip
            
            # Split into left and right parts
            left_part = image[:, :fold_line].copy()
            right_part = image[:, fold_line + fold_width:].copy()
            
            # Create new image by sliding pieces together
            new_image = np.zeros_like(image)
            
            # Place left part (unchanged)
            new_image[:, :fold_line] = left_part
            
            # Place right part shifted left (removing the fold strip)
            remaining_width = w - fold_line
            right_width = min(right_part.shape[1], remaining_width)
            if right_width > 0:
                new_image[:, fold_line:fold_line + right_width] = right_part[:, :right_width]
        
        # Apply slight blur at the seam for smoother transition
        if is_horizontal:
            seam_y = fold_line
            blur_region = min(5, seam_y, h - seam_y)
            if blur_region > 0:
                for y in range(max(0, seam_y - blur_region), min(h, seam_y + blur_region)):
                    kernel_size = 3
                    y_start = max(0, y - kernel_size // 2)
                    y_end = min(h, y + kernel_size // 2 + 1)
                    new_image[y, :] = np.mean(new_image[y_start:y_end, :], axis=0).astype(np.uint8)
        else:
            seam_x = fold_line
            blur_region = min(5, seam_x, w - seam_x)
            if blur_region > 0:
                for x in range(max(0, seam_x - blur_region), min(w, seam_x + blur_region)):
                    kernel_size = 3
                    x_start = max(0, x - kernel_size // 2)
                    x_end = min(w, x + kernel_size // 2 + 1)
                    new_image[:, x] = np.mean(new_image[:, x_start:x_end], axis=1).astype(np.uint8)
        
        return new_image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, number = self.samples[idx]
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            # Return black image if load fails
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        img = self._augment_image(img)
        
        # Resize to target size (if not already)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and apply normalization
        image_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize with ImageNet stats
        mean = torch.tensor(self.normalize['mean']).view(3, 1, 1)
        std = torch.tensor(self.normalize['std']).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Convert number to two digits (first digit and second digit)
        # For numbers 0-9, first digit is 0, second digit is the number
        # For numbers 10-99, first digit is tens place, second digit is ones place
        first_digit = number // 10
        second_digit = number % 10
        
        first_label = torch.tensor(first_digit, dtype=torch.long)
        second_label = torch.tensor(second_digit, dtype=torch.long)
        
        return image_tensor, (first_label, second_label)


class NumberClassifier(nn.Module):
    """Jersey number classifier using EfficientNet
    Predicts first digit and second digit separately (20 outputs total: 10 for each digit)
    """
    
    def __init__(self, model_name: str = "efficientnet_b0", freeze_backbone: bool = False):
        """
        Args:
            model_name: EfficientNet model name (e.g., efficientnet_b0, efficientnet_b1, etc.)
            freeze_backbone: Whether to freeze the vision encoder
        """
        super().__init__()
        
        # Load EfficientNet backbone (pretrained on ImageNet)
        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove the default classifier
            global_pool=''  # Get features before pooling
        )
        
        # Get feature dimension from the model
        # Try to get it from the model's num_features attribute first
        try:
            feature_dim = self.encoder.num_features
        except AttributeError:
            # Fallback: do a forward pass to determine feature dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                features = self.encoder.forward_features(dummy_input)
                
                # Handle different return types
                if isinstance(features, (list, tuple)):
                    feature_dim = features[-1].shape[1]
                elif len(features.shape) == 4:  # [B, C, H, W]
                    feature_dim = features.shape[1]
                elif len(features.shape) == 2:  # [B, C]
                    feature_dim = features.shape[1]
                else:
                    # Fallback: use common EfficientNet feature dimensions
                    if 'b0' in model_name.lower():
                        feature_dim = 1280
                    elif 'b1' in model_name.lower():
                        feature_dim = 1280
                    elif 'b2' in model_name.lower():
                        feature_dim = 1408
                    elif 'b3' in model_name.lower():
                        feature_dim = 1536
                    else:
                        feature_dim = 1280  # Default
        
        self.feature_dim = feature_dim
        print(f"Using EfficientNet model: {model_name}, feature dimension: {feature_dim}")
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared feature extractor
        self.shared_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Two separate classification heads: one for first digit (0-9), one for second digit (0-9)
        self.first_digit_classifier = nn.Linear(256, 10)  # 10 classes for first digit (0-9)
        self.second_digit_classifier = nn.Linear(256, 10)  # 10 classes for second digit (0-9)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Frozen backbone encoder")
    
    def forward(self, x):
        """Forward pass
        Returns:
            tuple: (first_digit_logits, second_digit_logits)
        """
        # Get features from EfficientNet
        features = self.encoder.forward_features(x)
        
        # Global average pooling to get fixed-size feature vector
        if len(features.shape) == 4:  # [B, C, H, W]
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)  # Flatten
        elif len(features.shape) == 2:  # Already flattened
            pass
        else:
            # Handle list/tuple of features
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if len(features.shape) == 4:
                features = self.global_pool(features)
                features = features.view(features.size(0), -1)
        
        # Pass through shared head
        shared_features = self.shared_head(features)
        
        # Get predictions for both digits
        first_digit_logits = self.first_digit_classifier(shared_features)
        second_digit_logits = self.second_digit_classifier(shared_features)
        
        return first_digit_logits, second_digit_logits


def compute_per_class_auroc(labels, probs):
    """Compute AUROC for each class (one-vs-rest) and return average over classes present
    
    Args:
        labels: Array of true labels (shape: [N])
        probs: Array of predicted probabilities (shape: [N, num_classes])
    
    Returns:
        Average AUROC across classes that have at least 2 samples, or 0.0 if not computable
    """
    if len(labels) == 0 or len(probs) == 0:
        return 0.0
    
    num_classes = probs.shape[1]
    per_class_aurocs = []
    
    for class_idx in range(num_classes):
        # Create binary labels for this class (1 if this class, 0 otherwise)
        binary_labels = (labels == class_idx).astype(int)
        
        # Skip if class has less than 2 samples (can't compute AUROC)
        if binary_labels.sum() < 2 or (1 - binary_labels).sum() < 2:
            continue
        
        try:
            # Get probabilities for this class
            class_probs = probs[:, class_idx]
            # Compute AUROC for this class (one-vs-rest)
            class_auroc = roc_auc_score(binary_labels, class_probs)
            per_class_aurocs.append(class_auroc)
        except (ValueError, IndexError):
            # Skip this class if AUROC can't be computed
            continue
    
    if len(per_class_aurocs) == 0:
        return 0.0
    
    # Return average AUROC across classes
    return np.mean(per_class_aurocs)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct_first = 0
    correct_second = 0
    correct_both = 0
    total = 0
    
    # Collect all predictions and labels for AUROC calculation
    all_first_probs = []
    all_first_labels = []
    all_second_probs = []
    all_second_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, (first_labels, second_labels) in pbar:
        images = images.to(device)
        first_labels = first_labels.to(device)
        second_labels = second_labels.to(device)
        
        optimizer.zero_grad()
        first_logits, second_logits = model(images)
        
        # Compute loss for both digits
        loss_first = criterion(first_logits, first_labels)
        loss_second = criterion(second_logits, second_labels)
        loss = loss_first + loss_second
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracies
        _, predicted_first = torch.max(first_logits.data, 1)
        _, predicted_second = torch.max(second_logits.data, 1)
        
        total += first_labels.size(0)
        correct_first += (predicted_first == first_labels).sum().item()
        correct_second += (predicted_second == second_labels).sum().item()
        correct_both += ((predicted_first == first_labels) & (predicted_second == second_labels)).sum().item()
        
        # Collect probabilities and labels for AUROC
        first_probs = F.softmax(first_logits, dim=1).detach().cpu().numpy()
        second_probs = F.softmax(second_logits, dim=1).detach().cpu().numpy()
        all_first_probs.append(first_probs)
        all_second_probs.append(second_probs)
        all_first_labels.append(first_labels.detach().cpu().numpy())
        all_second_labels.append(second_labels.detach().cpu().numpy())
        
        # Compute AUROC on accumulated data so far (for progress bar) - per class
        if len(all_first_probs) > 0:
            curr_first_probs = np.concatenate(all_first_probs, axis=0)
            curr_first_labels = np.concatenate(all_first_labels, axis=0)
            curr_auroc_first = compute_per_class_auroc(curr_first_labels, curr_first_probs)
        else:
            curr_auroc_first = 0.0
        
        if len(all_second_probs) > 0:
            curr_second_probs = np.concatenate(all_second_probs, axis=0)
            curr_second_labels = np.concatenate(all_second_labels, axis=0)
            curr_auroc_second = compute_per_class_auroc(curr_second_labels, curr_second_probs)
        else:
            curr_auroc_second = 0.0
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc1': 100 * correct_first / total,
            'acc2': 100 * correct_second / total,
            'acc_both': 100 * correct_both / total,
            'auroc1': f'{curr_auroc_first:.3f}',
            'auroc2': f'{curr_auroc_second:.3f}'
        })
    
    # Compute final AUROC using per-class approach
    all_first_probs = np.concatenate(all_first_probs, axis=0)
    all_first_labels = np.concatenate(all_first_labels, axis=0)
    all_second_probs = np.concatenate(all_second_probs, axis=0)
    all_second_labels = np.concatenate(all_second_labels, axis=0)
    
    # Compute per-class AUROC (one-vs-rest for each class, then average)
    auroc_first = compute_per_class_auroc(all_first_labels, all_first_probs)
    auroc_second = compute_per_class_auroc(all_second_labels, all_second_probs)
    
    avg_loss = total_loss / len(dataloader)
    acc_first = 100 * correct_first / total
    acc_second = 100 * correct_second / total
    acc_both = 100 * correct_both / total
    return avg_loss, (acc_first, acc_second, acc_both), (auroc_first, auroc_second)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct_first = 0
    correct_second = 0
    correct_both = 0
    total = 0
    
    # Collect all predictions and labels for AUROC calculation
    all_first_probs = []
    all_first_labels = []
    all_second_probs = []
    all_second_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, (first_labels, second_labels) in pbar:
            images = images.to(device)
            first_labels = first_labels.to(device)
            second_labels = second_labels.to(device)
            
            first_logits, second_logits = model(images)
            
            # Compute loss for both digits
            loss_first = criterion(first_logits, first_labels)
            loss_second = criterion(second_logits, second_labels)
            loss = loss_first + loss_second
            
            total_loss += loss.item()
            
            # Compute accuracies
            _, predicted_first = torch.max(first_logits.data, 1)
            _, predicted_second = torch.max(second_logits.data, 1)
            
            total += first_labels.size(0)
            correct_first += (predicted_first == first_labels).sum().item()
            correct_second += (predicted_second == second_labels).sum().item()
            correct_both += ((predicted_first == first_labels) & (predicted_second == second_labels)).sum().item()
            
            # Collect probabilities and labels for AUROC
            first_probs = F.softmax(first_logits, dim=1).detach().cpu().numpy()
            second_probs = F.softmax(second_logits, dim=1).detach().cpu().numpy()
            all_first_probs.append(first_probs)
            all_second_probs.append(second_probs)
            all_first_labels.append(first_labels.detach().cpu().numpy())
            all_second_labels.append(second_labels.detach().cpu().numpy())
            
            # Compute AUROC on accumulated data so far (for progress bar) - per class
            if len(all_first_probs) > 0:
                curr_first_probs = np.concatenate(all_first_probs, axis=0)
                curr_first_labels = np.concatenate(all_first_labels, axis=0)
                curr_auroc_first = compute_per_class_auroc(curr_first_labels, curr_first_probs)
            else:
                curr_auroc_first = 0.0
            
            if len(all_second_probs) > 0:
                curr_second_probs = np.concatenate(all_second_probs, axis=0)
                curr_second_labels = np.concatenate(all_second_labels, axis=0)
                curr_auroc_second = compute_per_class_auroc(curr_second_labels, curr_second_probs)
            else:
                curr_auroc_second = 0.0
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc1': 100 * correct_first / total,
                'acc2': 100 * correct_second / total,
                'acc_both': 100 * correct_both / total,
                'auroc1': f'{curr_auroc_first:.3f}',
                'auroc2': f'{curr_auroc_second:.3f}'
            })
    
    # Compute final AUROC using per-class approach
    all_first_probs = np.concatenate(all_first_probs, axis=0)
    all_first_labels = np.concatenate(all_first_labels, axis=0)
    all_second_probs = np.concatenate(all_second_probs, axis=0)
    all_second_labels = np.concatenate(all_second_labels, axis=0)
    
    # Compute per-class AUROC (one-vs-rest for each class, then average)
    auroc_first = compute_per_class_auroc(all_first_labels, all_first_probs)
    auroc_second = compute_per_class_auroc(all_second_labels, all_second_probs)
    
    avg_loss = total_loss / len(dataloader)
    acc_first = 100 * correct_first / total
    acc_second = 100 * correct_second / total
    acc_both = 100 * correct_both / total
    return avg_loss, (acc_first, acc_second, acc_both), (auroc_first, auroc_second)


def main():
    parser = argparse.ArgumentParser(description="Train jersey number classifier")
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet_b0',
        help='EfficientNet model name (default: efficientnet_b0). Options: efficientnet_b0, efficientnet_b1, efficientnet_b2, etc.'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze the vision encoder backbone'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='runs/jersey_classifier',
        help='Output directory for checkpoints (default: runs/jersey_classifier)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='jersey_number_dataset',
        help='Path to combined dataset directory (with train/val/test subdirs) (default: jersey_number_dataset)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (number of epochs without improvement) (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("  Run combine_jersey_datasets.py first to create the combined dataset")
        return
    
    print(f"Using dataset directory: {dataset_dir.absolute()}")
    
    # Create datasets for each split
    print("\nLoading datasets...")
    train_dataset = JerseyNumberDataset(
        dataset_dir=dataset_dir,
        split="train",
        image_size=args.image_size,
        augment=True
    )
    
    val_dataset = JerseyNumberDataset(
        dataset_dir=dataset_dir,
        split="val",
        image_size=args.image_size,
        augment=False
    )
    
    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        return
    
    if len(val_dataset) == 0:
        print("Warning: No validation samples found!")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    print("  Architecture: Two-digit prediction (10 classes for first digit, 10 for second digit)")
    model = NumberClassifier(
        model_name=args.model,
        freeze_backbone=args.freeze_backbone
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Early stopping patience: {args.patience} epochs")
    
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_auroc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_auroc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print results
        train_acc_first, train_acc_second, train_acc_both = train_acc
        val_acc_first, val_acc_second, val_acc_both = val_acc
        train_auroc_first, train_auroc_second = train_auroc
        val_auroc_first, val_auroc_second = val_auroc
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Train Acc - First: {train_acc_first:.2f}%, Second: {train_acc_second:.2f}%, Both: {train_acc_both:.2f}%")
        print(f"    Train AUROC - First: {train_auroc_first:.4f}, Second: {train_auroc_second:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    Val Acc - First: {val_acc_first:.2f}%, Second: {val_acc_second:.2f}%, Both: {val_acc_both:.2f}%")
        print(f"    Val AUROC - First: {val_auroc_first:.4f}, Second: {val_auroc_second:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc_first': val_acc_first,
                'val_acc_second': val_acc_second,
                'val_acc_both': val_acc_both,
                'val_auroc_first': val_auroc_first,
                'val_auroc_second': val_auroc_second,
                'train_loss': train_loss,
                'train_acc_first': train_acc_first,
                'train_acc_second': train_acc_second,
                'train_acc_both': train_acc_both,
                'train_auroc_first': train_auroc_first,
                'train_auroc_second': train_auroc_second,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 and checkpoint is not None:
            checkpoint['epoch'] = epoch
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

