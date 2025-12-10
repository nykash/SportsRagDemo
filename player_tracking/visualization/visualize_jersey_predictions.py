#!/usr/bin/env python3
"""
Create a 5x5 grid collage showing jersey number predictions from the classifier model.
Shows ground truth vs predicted labels with confidence scores.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFont

# Import NumberClassifier from training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_jersey_number_classifier import NumberClassifier

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path=None, model_name="efficientnet_b0", image_size=224):
    """Load jersey number classifier model from checkpoint."""
    print(f"Loading jersey number classifier model from {checkpoint_path}...")
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = "runs/jersey_classifier/best_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Create model architecture
    model = NumberClassifier(model_name=model_name, freeze_backbone=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    if 'val_acc_both' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc_both']:.2f}%")
    
    return model, image_size


def preprocess_image(image_path, image_size=224):
    """Preprocess image to match training pipeline."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and apply ImageNet normalization
    image_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0), img  # Return both tensor and original image


def predict_number(model, image_tensor):
    """Predict jersey number from preprocessed image tensor."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        first_logits, second_logits = model(image_tensor)
        
        # Get probabilities
        first_probs = F.softmax(first_logits, dim=1)
        second_probs = F.softmax(second_logits, dim=1)
        
        # Get predicted digits
        first_digit = torch.argmax(first_probs, dim=1).item()
        second_digit = torch.argmax(second_probs, dim=1).item()
        
        # Get confidence (probability of predicted class)
        first_confidence = first_probs[0, first_digit].item()
        second_confidence = second_probs[0, second_digit].item()
        
        # Combine digits to get jersey number
        jersey_number = first_digit * 10 + second_digit
        
        # Overall confidence is the product of both digit confidences
        confidence = first_confidence * second_confidence
        
        return jersey_number, confidence, first_confidence, second_confidence


def extract_number_from_filename(filename):
    """Extract jersey number from filename."""
    # Format: "xxx_num14.jpg" or "xxx_num24.jpg"
    filename = Path(filename).stem
    
    # Try pattern: _numXX
    if '_num' in filename:
        try:
            num_str = filename.split('_num')[-1]
            number = int(num_str)
            if 0 <= number <= 99:
                return number
        except (ValueError, IndexError):
            pass
    
    # If not found, try to parse from end of filename
    parts = filename.split('_')
    for part in reversed(parts):
        try:
            number = int(part)
            if 0 <= number <= 99:
                return number
        except ValueError:
            continue
    
    return None


def load_samples_from_dataset(dataset_dir, split="val", num_samples=25):
    """Load sample images from dataset."""
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Find all images
    image_files = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
    
    # Extract samples with valid numbers
    samples = []
    for img_path in image_files:
        number = extract_number_from_filename(img_path.name)
        if number is not None:
            samples.append((img_path, number))
    
    # Randomly sample
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples


def create_collage(samples_data, output_path, image_size=224, grid_size=5):
    """Create a grid collage showing predictions."""
    num_samples = len(samples_data)
    
    # Calculate grid dimensions (make it as square as possible)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Calculate cell size (image + padding + text)
    cell_height = image_size + 80  # Extra space for text
    cell_width = image_size
    padding = 10
    border_width = 3
    
    # Create canvas
    canvas_width = grid_size * cell_width + (grid_size + 1) * padding
    canvas_height = grid_size * cell_height + (grid_size + 1) * padding
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    
    # Try to load a font
    try:
        # Try to use a larger font
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw each sample (only draw up to num_samples)
    for idx, (image, true_label, pred_label, confidence, first_conf, second_conf) in enumerate(samples_data):
        if idx >= grid_size * grid_size:
            break
        row = idx // grid_size
        col = idx % grid_size
        
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        
        # Convert numpy image to PIL
        if isinstance(image, np.ndarray):
            # Denormalize if needed (assuming it's in [0, 1])
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # Ensure RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Resize image if needed
        if pil_image.size != (image_size, image_size):
            pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        # Paste image
        canvas.paste(pil_image, (x, y))
        
        # Determine if prediction is correct
        is_correct = (true_label == pred_label)
        border_color = (0, 200, 0) if is_correct else (200, 0, 0)  # Green or red
        
        # Draw border
        draw = ImageDraw.Draw(canvas)
        for i in range(border_width):
            draw.rectangle(
                [x - i, y - i, x + image_size + i, y + image_size + i],
                outline=border_color,
                width=1
            )
        
        # Draw text below image
        text_y = y + image_size + 5
        
        # Ground truth label
        gt_text = f"GT: {true_label:02d}"
        draw.text((x, text_y), gt_text, fill=(0, 0, 0), font=font_large)
        
        # Predicted label
        pred_text = f"Pred: {pred_label:02d}"
        pred_color = (0, 150, 0) if is_correct else (200, 0, 0)
        draw.text((x, text_y + 25), pred_text, fill=pred_color, font=font_large)
        
        # Confidence
        conf_text = f"Conf: {confidence:.2f}"
        draw.text((x, text_y + 50), conf_text, fill=(100, 100, 100), font=font_small)
    
    # Save collage
    canvas.save(output_path)
    print(f"Collage saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create collage of jersey number predictions")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='runs/jersey_classifier/best_model.pt',
        help='Path to model checkpoint (default: runs/jersey_classifier/best_model.pt)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet_b0',
        help='EfficientNet model name (default: efficientnet_b0)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='jersey_number_dataset',
        help='Path to dataset directory (default: jersey_number_dataset)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to use (default: val)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=25,
        help='Number of samples to show (default: 25 for 5x5 grid)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='jersey_predictions_collage.png',
        help='Output path for collage image (default: jersey_predictions_collage.png)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    model, image_size = load_model(args.checkpoint, args.model, args.image_size)
    
    # Load samples
    print(f"\nLoading samples from {args.split} split...")
    samples = load_samples_from_dataset(args.dataset_dir, args.split, args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("Error: No valid samples found!")
        return
    
    # Process samples and make predictions
    print("\nMaking predictions...")
    samples_data = []
    correct_count = 0
    
    for img_path, true_label in samples:
        # Preprocess image
        image_tensor, original_image = preprocess_image(img_path, args.image_size)
        if image_tensor is None:
            continue
        
        # Make prediction
        pred_label, confidence, first_conf, second_conf = predict_number(model, image_tensor)
        
        # Check if correct
        if true_label == pred_label:
            correct_count += 1
        
        samples_data.append((original_image, true_label, pred_label, confidence, first_conf, second_conf))
        
        print(f"  {img_path.name}: GT={true_label:02d}, Pred={pred_label:02d}, Conf={confidence:.3f} {'✓' if true_label == pred_label else '✗'}")
    
    print(f"\nAccuracy: {correct_count}/{len(samples_data)} ({100*correct_count/len(samples_data):.1f}%)")
    
    # Create collage
    print("\nCreating collage...")
    grid_size = int(np.ceil(np.sqrt(args.num_samples)))
    create_collage(samples_data, args.output, args.image_size, grid_size=grid_size)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

