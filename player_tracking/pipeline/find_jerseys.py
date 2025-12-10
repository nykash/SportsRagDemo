import os
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor
import glob
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
from tqdm import tqdm

video_paths = os.listdir("data/splices")
video_paths = [path for path in video_paths if path[0] not in ["1", "4"]]
video_paths = [os.path.join("data/splices", video_path) for video_path in video_paths if video_path.endswith(".mp4") and "1" in video_path.split("_")[-1]]

# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

def save_tracking_video(outputs_per_frame, video_frames, output_path="sam3_tracking_output.mp4", fps=30, alpha=0.5):
    """
    Save tracking outputs as an MP4 video by overlaying masks directly on frames.
    
    Args:
        outputs_per_frame: Dict {frame_idx: {obj_id: mask_tensor}} after prepare_masks_for_visualization
        video_frames: List of video frames (RGB numpy arrays)
        output_path: Output video file path
        fps: Frames per second for output video
        alpha: Transparency for mask overlay (0-1)
    """
    from sam3.visualization_utils import COLORS, load_frame
    
    # Get frame dimensions from first frame
    first_frame = load_frame(video_frames[0])
    if first_frame.dtype == np.float32 or first_frame.max() <= 1.0:
        first_frame = (first_frame * 255).astype(np.uint8)
    height, width = first_frame.shape[:2]

    # Validate dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame dimensions: width={width}, height={height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_indices = sorted(outputs_per_frame.keys())
    for frame_idx in tqdm(frame_indices):
        # Load and prepare frame
        frame = load_frame(video_frames[frame_idx])
        if frame.dtype == np.float32 or frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        frame = frame[..., :3]  # Ensure RGB
        overlay = frame.copy()
        
        # Overlay masks for this frame
        if frame_idx in outputs_per_frame:
            for obj_id, binary_mask in outputs_per_frame[frame_idx].items():
                # Convert mask to numpy if needed
                if hasattr(binary_mask, 'numpy'):
                    mask = binary_mask.numpy()
                elif hasattr(binary_mask, 'cpu'):
                    mask = binary_mask.cpu().numpy()
                else:
                    mask = np.array(binary_mask)
                
                # Ensure mask is 2D
                if mask.ndim > 2:
                    mask = mask.squeeze()
                if mask.ndim != 2:
                    print(f"Warning: Skipping invalid mask with shape {mask.shape} for obj_id {obj_id}")
                    continue
                
                # Skip empty masks
                if mask.size == 0:
                    print(f"Warning: Skipping empty mask for obj_id {obj_id}")
                    continue
                
                # Resize mask if needed
                if mask.shape != (height, width):
                    # Ensure we have valid dimensions for resize
                    if width > 0 and height > 0:
                        mask = cv2.resize(mask.astype(np.float32), (width, height), 
                                        interpolation=cv2.INTER_NEAREST)
                    else:
                        print(f"Warning: Skipping resize due to invalid dimensions: width={width}, height={height}")
                        continue
                
                # Get color for this object
                # Convert obj_id to integer for indexing, handling both int and string types
                if isinstance(obj_id, (int, np.integer)):
                    obj_id_int = int(obj_id)
                else:
                    obj_id_int = abs(hash(str(obj_id)))
                color = COLORS[obj_id_int % len(COLORS)]
                color255 = (color * 255).astype(np.uint8)
                
                # Apply mask overlay
                mask_bool = mask > 0.5
                for c in range(3):
                    overlay[..., c][mask_bool] = (
                        alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
                    ).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {output_path} ({len(frame_indices)} frames at {fps} fps)")

def propagate_in_video(predictor, session_id, max_frames=None):
    # we will just propagate from frame 0 to the end of the video (or max_frames if specified)
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_index = response["frame_index"]
        if max_frames is not None and frame_index >= max_frames:
            break
        outputs_per_frame[frame_index] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def extract_quadrilateral_from_mask(mask):
    """Extract quadrilateral (4 corner points) from a binary mask.
    
    Args:
        mask: Binary mask (numpy array)
    
    Returns:
        List of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] or None if mask is empty
        Points are ordered in a consistent manner (e.g., top-left, top-right, bottom-right, bottom-left)
    """
    if mask.size == 0:
        return None
    
    # Convert mask to uint8 binary format
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    # epsilon is the maximum distance from the original contour to the approximated contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have exactly 4 points, use them
    if len(approx) == 4:
        # Convert to list of [x, y] coordinates
        quad = [[int(point[0][0]), int(point[0][1])] for point in approx]
    else:
        # If not 4 points, use minimum area rectangle and get its 4 corners
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        # Convert to list of [x, y] coordinates and ensure integer values
        quad = [[int(point[0]), int(point[1])] for point in box]
    
    # Sort points in a consistent order: top-left, top-right, bottom-right, bottom-left
    # First, sort by y-coordinate to separate top and bottom
    quad_sorted = sorted(quad, key=lambda p: p[1])
    top_points = sorted(quad_sorted[:2], key=lambda p: p[0])  # Sort top points by x
    bottom_points = sorted(quad_sorted[2:], key=lambda p: p[0])  # Sort bottom points by x
    
    # Return: top-left, top-right, bottom-right, bottom-left
    return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

def extract_quadrilaterals_from_outputs(outputs_per_frame):
    """Extract quadrilaterals (4 corner points) for all objects in all frames.
    
    Args:
        outputs_per_frame: Dict {frame_idx: {obj_id: mask_tensor}} after prepare_masks_for_visualization
    
    Returns:
        Dict {frame_idx: {obj_id: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}}
    """
    quadrilaterals_per_frame = {}
    
    for frame_idx, objects in outputs_per_frame.items():
        quadrilaterals_per_frame[frame_idx] = {}
        for obj_id, binary_mask in objects.items():
            # Convert mask to numpy if needed
            if hasattr(binary_mask, 'numpy'):
                mask = binary_mask.numpy()
            elif hasattr(binary_mask, 'cpu'):
                mask = binary_mask.cpu().numpy()
            else:
                mask = np.array(binary_mask)
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim != 2:
                continue
            
            # Extract quadrilateral
            quad = extract_quadrilateral_from_mask(mask)
            if quad is not None:
                # Convert obj_id to string for JSON serialization
                obj_id_str = str(obj_id) if not isinstance(obj_id, str) else obj_id
                quadrilaterals_per_frame[frame_idx][obj_id_str] = quad
    
    return quadrilaterals_per_frame

def save_quadrilaterals_to_json(quadrilaterals_per_frame, output_path):
    """Save quadrilaterals to a JSON file.
    
    Args:
        quadrilaterals_per_frame: Dict {frame_idx: {obj_id: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}}
        output_path: Path to output JSON file
    """
    # Convert frame indices to strings for JSON serialization
    json_data = {str(frame_idx): quads for frame_idx, quads in quadrilaterals_per_frame.items()}
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Quadrilaterals saved to {output_path}")

def run_predictor(predictor, session_id, prompt_text_str, max_frames=None):
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )

    outputs_per_frame = propagate_in_video(predictor, session_id, max_frames=max_frames)

    return outputs_per_frame


gpus_to_use = range(torch.cuda.device_count())

# Create output directories
output_video_dir = "output_videos"
output_quad_dir = "output_quadrilaterals"
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_quad_dir, exist_ok=True)

# Build predictor once (like in the notebook)
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

for i, video_path in enumerate(video_paths):
    print(f"Processing video {i+1} of {len(video_paths)}: {video_path}")
    
    # Get base filename for output files
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_path = os.path.join(output_video_dir, f"sam3_tracking_output_{video_basename}.mp4")
    quad_output_path = os.path.join(output_quad_dir, f"quadrilaterals_{video_basename}.json")
    
    # Skip if already processed
    if os.path.exists(video_output_path) and os.path.exists(quad_output_path):
        print(f"Skipping {video_path} - already processed")
        continue
    
    # Check frame count and skip if more than 500 frames
    MAX_FRAMES = 500
    fps = 30  # Default FPS for image sequences or if unable to get FPS from video
    total_frames = 0
    
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if unable to get FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        video_frames_list = glob.glob(os.path.join(video_path, "*.jpg"))
        total_frames = len(video_frames_list)
    
    if total_frames > MAX_FRAMES:
        print(f"Skipping {video_path} - has {total_frames} frames (more than {MAX_FRAMES})")
        continue
    
    # Start session for this video
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    # Load video frames
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()
    
    # Run predictor
    outputs_per_frame = run_predictor(predictor, session_id, "number")
    
    # Prepare masks for visualization (critical step from notebook)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
    
    # Extract quadrilaterals
    quadrilaterals_per_frame = extract_quadrilaterals_from_outputs(outputs_per_frame)
    
    # Save tracking video to output folder
    save_tracking_video(outputs_per_frame, video_frames_for_vis, fps=fps, output_path=video_output_path)
    
    # Save quadrilaterals to JSON in output folder
    save_quadrilaterals_to_json(quadrilaterals_per_frame, quad_output_path)
    
    # Close session for this video
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    torch.cuda.empty_cache()

# After all inference is done, shutdown the predictor
# to free up the multi-GPU process group
predictor.shutdown()
    
