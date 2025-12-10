import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import glob
import sys

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
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, image_size

def normalize_quad_points(quad):
    """Normalize quadrilateral points to consistent order: top-left, top-right, bottom-right, bottom-left.
    
    This ensures the extracted region has the correct rotation/orientation.
    
    Args:
        quad: List of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    Returns:
        Ordered quadrilateral points: [top-left, top-right, bottom-right, bottom-left]
    """
    if len(quad) != 4:
        return quad
    
    # Convert to numpy array for easier manipulation
    points = np.array(quad, dtype=np.float32)
    
    # Sort by y-coordinate first to separate top and bottom
    # This works well for most cases where the quad is roughly upright
    sorted_by_y = sorted(points, key=lambda p: p[1])
    top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])  # Sort top points by x (left to right)
    bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])  # Sort bottom points by x (left to right)
    
    # Return: top-left, top-right, bottom-right, bottom-left
    ordered = [
        top_points[0].tolist(),      # top-left
        top_points[1].tolist(),      # top-right
        bottom_points[1].tolist(),   # bottom-right
        bottom_points[0].tolist()    # bottom-left
    ]
    
    return ordered


def extract_jersey_region(frame, quad, expand_factor=0.15):
    """Extract jersey region from frame using quadrilateral coordinates with proper rotation.
    
    Args:
        frame: Input frame (RGB)
        quad: Quadrilateral coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        expand_factor: Factor to expand the bounding box (default: 0.15 = 15%)
    
    Returns:
        Extracted ROI region with correct rotation
    """
    # Normalize point order: top-left, top-right, bottom-right, bottom-left
    quad_ordered = normalize_quad_points(quad)
    quad_np = np.array(quad_ordered, dtype=np.float32)
    
    # Calculate bounding box dimensions for output size
    x_coords = quad_np[:, 0]
    y_coords = quad_np[:, 1]
    
    # Calculate width and height from the quadrilateral
    width_top = np.linalg.norm(quad_np[1] - quad_np[0])
    width_bottom = np.linalg.norm(quad_np[2] - quad_np[3])
    height_left = np.linalg.norm(quad_np[3] - quad_np[0])
    height_right = np.linalg.norm(quad_np[2] - quad_np[1])
    
    # Use average dimensions
    avg_width = (width_top + width_bottom) / 2.0
    avg_height = (height_left + height_right) / 2.0
    
    # Expand dimensions
    output_width = int(avg_width * (1.0 + expand_factor))
    output_height = int(avg_height * (1.0 + expand_factor))
    
    # Expand the quadrilateral outward from center
    center = np.mean(quad_np, axis=0)
    expanded_quad = quad_np.copy()
    for i in range(4):
        # Expand outward from center
        direction = quad_np[i] - center
        expanded_quad[i] = center + direction * (1.0 + expand_factor)
    
    # Clip to frame boundaries
    frame_height, frame_width = frame.shape[:2]
    expanded_quad[:, 0] = np.clip(expanded_quad[:, 0], 0, frame_width - 1)
    expanded_quad[:, 1] = np.clip(expanded_quad[:, 1], 0, frame_height - 1)
    
    # Define destination points for perspective transform (rectified rectangle)
    # Output will be a rectangle with correct orientation
    dst_points = np.array([
        [0, 0],                          # top-left
        [output_width - 1, 0],           # top-right
        [output_width - 1, output_height - 1],  # bottom-right
        [0, output_height - 1]           # bottom-left
    ], dtype=np.float32)
    
    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(expanded_quad, dst_points)
    
    # Apply perspective transformation to extract the region
    roi = cv2.warpPerspective(frame, M, (output_width, output_height), 
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    
    if roi.size == 0 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
        return None
    
    return roi

def preprocess_image(roi, image_size=224):
    """Preprocess ROI image to match training pipeline exactly (without augmentation).
    
    Args:
        roi: RGB image array (already in RGB format from extract_jersey_region)
        image_size: Target image size
    
    Returns:
        Preprocessed tensor ready for model input
    """
    if roi is None or roi.size == 0:
        return None
    
    # Ensure RGB format (handle different input formats)
    if len(roi.shape) == 2:
        # Grayscale - convert to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    elif roi.shape[2] == 4:
        # RGBA - convert to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
    elif roi.shape[2] == 3:
        # Already 3 channels - should already be RGB (frame is converted before extraction)
        pass
    
    # Resize to target size (match training: uses cv2.resize)
    if roi.shape[0] != image_size or roi.shape[1] != image_size:
        roi = cv2.resize(roi, (image_size, image_size))
    
    # Normalize to [0, 1] - match training exactly
    roi = roi.astype(np.float32) / 255.0
    
    # Convert to tensor and apply normalization - match training exactly
    image_tensor = torch.from_numpy(roi).permute(2, 0, 1)  # HWC -> CHW
    
    # ImageNet normalization - match training exactly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)  # Add batch dimension


def recognize_number(model, roi, image_size=224):
    """Recognize jersey number from ROI using the classifier model.
    
    Returns:
        tuple: (jersey_number, confidence, first_probs, second_probs)
               where first_probs and second_probs are probability distributions
    """
    if roi is None or roi.size == 0:
        return None, 0.0, None, None
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(roi, image_size)
        if image_tensor is None:
            return None, 0.0, None, None
        
        image_tensor = image_tensor.to(device)
        
        # Get predictions
        with torch.no_grad():
            first_logits, second_logits = model(image_tensor)
            
            # Get probabilities
            first_probs = F.softmax(first_logits, dim=1).cpu().numpy()[0]  # Shape: (10,)
            second_probs = F.softmax(second_logits, dim=1).cpu().numpy()[0]  # Shape: (10,)
            
            # Get predicted digits
            first_digit = np.argmax(first_probs)
            second_digit = np.argmax(second_probs)
            
            # Get confidence (probability of predicted class)
            first_confidence = first_probs[first_digit]
            second_confidence = second_probs[second_digit]
            
            # Combine digits to get jersey number
            jersey_number = first_digit * 10 + second_digit
            
            # Overall confidence is the product of both digit confidences
            confidence = first_confidence * second_confidence
            
            return jersey_number, confidence, first_probs, second_probs
        
    except Exception as e:
        # If recognition fails, return None
        return None, 0.0, None, None

def normalize_quadrilateral(quad, img_width, img_height):
    """Normalize quadrilateral coordinates to [0, 1] range."""
    normalized = []
    for point in quad:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        normalized.append([x_norm, y_norm])
    return normalized

def find_video_file(quad_filename):
    """Find corresponding video file for a quadrilateral JSON file."""
    # Extract video ID and timestamps from filename
    # Format: quadrilaterals_{video_id}_{start}_{end}_{something}.json
    basename = os.path.basename(quad_filename)
    parts = basename.replace("quadrilaterals_", "").replace(".json", "").split("_")
    
    if len(parts) >= 3:
        video_id = parts[0]
        start_time = parts[1]
        end_time = parts[2]
        suffix = parts[3] if len(parts) > 3 else "1"
        
        # Try to find matching video file
        video_pattern = f"data/splices/{video_id}_{start_time}_{end_time}_{suffix}.mp4"
        if os.path.exists(video_pattern):
            return video_pattern
        
        # Try without suffix
        video_pattern = f"data/splices/{video_id}_{start_time}_{end_time}.mp4"
        if os.path.exists(video_pattern):
            return video_pattern
    
    return None

def process_quadrilaterals_file(quad_file, model, image_size, output_dir="output_quadrilaterals_with_jerseys"):
    """Process a single quadrilaterals JSON file and recognize jersey numbers."""
    print(f"Processing {quad_file}...")

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(quad_file).replace("quadrilaterals_", "quadrilaterals_with_jerseys_")
    if os.path.exists(os.path.join(output_dir, output_filename)):
        #print(f"  Warning: {output_filename} already exists")
        return None
    
    # Load quadrilaterals
    with open(quad_file, 'r') as f:
        quadrilaterals = json.load(f)
    
    # Find corresponding video file
    video_path = find_video_file(quad_file)
    if video_path is None:
        print(f"  Warning: Could not find video file for {quad_file}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: Could not open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Track unique object IDs and their recognized numbers
    obj_id_to_jersey = {}  # {obj_id: (jersey_number, confidence, frame_idx)}
    obj_id_samples = defaultdict(list)  # {obj_id: [(frame_idx, quad, roi)]}
    
    # Collect samples for each unique object ID
    for frame_idx_str, objects in quadrilaterals.items():
        frame_idx = int(frame_idx_str)
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for obj_id, quad in objects.items():
            # Extract jersey region
            roi = extract_jersey_region(frame_rgb, quad)
            if roi is not None and roi.size > 0:
                obj_id_samples[obj_id].append((frame_idx, quad, roi))
    
    cap.release()
    
    # Recognize numbers for each unique object ID using all frames
    # Accumulate probabilities across all frames and pick the class with highest probability
    for obj_id, samples in obj_id_samples.items():
        if len(samples) == 0:
            continue
        
        # Accumulate probabilities across all frames
        all_first_probs = []  # List of probability distributions for first digit
        all_second_probs = []  # List of probability distributions for second digit
        
        # Process every frame
        for frame_idx, quad, roi in samples:
            jersey_num, confidence, first_probs, second_probs = recognize_number(model, roi, image_size)
            
            if first_probs is not None and second_probs is not None:
                all_first_probs.append(first_probs)
                all_second_probs.append(second_probs)
        
        if len(all_first_probs) == 0:
            continue
        
        # Average probabilities across all frames (better than max for robustness)
        avg_first_probs = np.mean(all_first_probs, axis=0)  # Shape: (10,)
        avg_second_probs = np.mean(all_second_probs, axis=0)  # Shape: (10,)
        
        # Pick the class with highest probability
        first_digit = np.argmax(avg_first_probs)
        second_digit = np.argmax(avg_second_probs)
        
        # Get confidence from averaged probabilities
        first_confidence = avg_first_probs[first_digit]
        second_confidence = avg_second_probs[second_digit]
        
        # Combine digits to get jersey number
        jersey_num = first_digit * 10 + second_digit
        
        # Overall confidence is the product of both digit confidences
        confidence = first_confidence * second_confidence
        
        # Store the result
        if confidence > 0.01:  # Very low threshold since we're averaging
            obj_id_to_jersey[obj_id] = (jersey_num, confidence)
    
    # Create output with normalized coordinates and jersey number assignments
    output_quadrilaterals = {}
    for frame_idx_str, objects in quadrilaterals.items():
        output_quadrilaterals[frame_idx_str] = {}
        for obj_id, quad in objects.items():
            # Normalize coordinates
            normalized_quad = normalize_quadrilateral(quad, width, height)
            
            # Assign jersey number if recognized, otherwise use original ID with question mark
            if obj_id in obj_id_to_jersey:
                jersey_num, conf = obj_id_to_jersey[obj_id]
                # Use jersey number as key if confidence is high enough
                if conf > 0.2:
                    output_key = str(jersey_num)
                else:
                    output_key = f"{obj_id}?"
            else:
                output_key = f"{obj_id}?"
            
            output_quadrilaterals[frame_idx_str][output_key] = normalized_quad
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(quad_file).replace("quadrilaterals_", "quadrilaterals_with_jerseys_")
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(output_quadrilaterals, f, indent=2)
    
    print(f"  Saved to {output_path}")
    print(f"  Recognized {len([v for v in obj_id_to_jersey.values() if v[1] > 0.2])} jersey numbers")
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognize jersey numbers using trained classifier")
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
        '--quad-dir',
        type=str,
        default='output_quadrilaterals',
        help='Directory containing quadrilateral JSON files (default: output_quadrilaterals)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_quadrilaterals_with_jerseys',
        help='Output directory for results (default: output_quadrilaterals_with_jerseys)'
    )
    
    args = parser.parse_args()
    
    # Load classifier model
    model, image_size = load_model(args.checkpoint, args.model, args.image_size)
    
    # Get all quadrilateral files
    quad_dir = args.quad_dir
    quad_files = glob.glob(os.path.join(quad_dir, "quadrilaterals_*.json"))
    
    print(f"Found {len(quad_files)} quadrilateral files")
    
    # Process each file
    for quad_file in tqdm(quad_files, desc="Processing files"):
        try:
            process_quadrilaterals_file(quad_file, model, image_size, args.output_dir)
        except Exception as e:
            print(f"Error processing {quad_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Done!")

if __name__ == "__main__":
    main()

