import ultralytics
import json
import cv2
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from shapely.geometry import Polygon, box
from tqdm import tqdm

@dataclass
class PlayerAction:
    action: str
    confidence: float
    start_time: float
    end_time: float
    jersey_number: Optional[str] = None
    position: Optional[Tuple[float, float]] = None  # Normalized center coordinates (x, y) in [0, 1]
    team: Optional[str] = None  # Team assignment: "team_A" or "team_B"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to Python native types for JSON serialization
        position = None
        if self.position:
            position = [float(x) for x in self.position]
        
        return {
            'action': self.action,
            'confidence': float(self.confidence),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'jersey_number': self.jersey_number,
            'position': position,
            'team': self.team
        }

@dataclass
class StatsOverClip:
    actions: List[PlayerAction]
    start_time: float
    end_time: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'actions': [action.to_dict() for action in self.actions],
            'start_time': self.start_time,
            'end_time': self.end_time
        }

player_tracking_model = ultralytics.YOLO("yolo11m-seg.pt")  # Segmentation model for better tracking
action_model = ultralytics.YOLO("train4/weights/best.pt")


def bbox_to_polygon(bbox_xyxy: np.ndarray, img_width: int, img_height: int) -> Polygon:
    """Convert bounding box (xyxy format) to Shapely Polygon.
    
    Args:
        bbox_xyxy: Bounding box in format [x1, y1, x2, y2]
        img_width: Image width (for normalization if needed)
        img_height: Image height (for normalization if needed)
    
    Returns:
        Shapely Polygon
    """
    x1, y1, x2, y2 = bbox_xyxy
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    return box(x1, y1, x2, y2)


def quad_to_polygon(quad: List[List[float]], img_width: int, img_height: int) -> Polygon:
    """Convert normalized quadrilateral to Shapely Polygon.
    
    Args:
        quad: Quadrilateral points in normalized coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        img_width: Image width to denormalize coordinates
        img_height: Image height to denormalize coordinates
    
    Returns:
        Shapely Polygon
    """
    # Denormalize coordinates
    denormalized = []
    for point in quad:
        x, y = point
        denormalized.append([x * img_width, y * img_height])
    return Polygon(denormalized)


def check_intersection(bbox_poly: Polygon, quad_poly: Polygon, iou_threshold: float = 0.05) -> bool:
    """Check if bounding box and quadrilateral intersect.
    
    Uses multiple criteria:
    1. Checks if quad is fully contained in bbox (most common case - jersey inside player)
    2. Checks if >50% of quad area is covered by bbox
    3. Checks IoU
    4. Checks if center of quad is inside bbox (fallback)
    
    Args:
        bbox_poly: Shapely Polygon for bounding box
        quad_poly: Shapely Polygon for quadrilateral
        iou_threshold: Minimum IoU to consider as intersection (lowered for better matching)
    
    Returns:
        True if intersection is significant
    """
    if not bbox_poly.is_valid or not quad_poly.is_valid:
        return False
    
    # Fast check: if quad is fully contained in bbox, it's definitely a match
    if bbox_poly.contains(quad_poly):
        return True
    
    # Check if center of quad is inside bbox (quick check for partial overlap)
    quad_center = quad_poly.centroid
    if bbox_poly.contains(quad_center):
        # If center is inside, calculate detailed intersection
        intersection = bbox_poly.intersection(quad_poly)
        if not intersection.is_empty:
            intersection_area = intersection.area
            quad_area = quad_poly.area
            
            if quad_area > 0:
                # If >30% of quad is covered, it's a match
                coverage = intersection_area / quad_area
                if coverage > 0.3:
                    return True
            
            # Also check IoU
            union_area = bbox_poly.union(quad_poly).area
            if union_area > 0:
                iou = intersection_area / union_area
                if iou >= iou_threshold:
                    return True
    
    # Last check: general intersection with reasonable overlap
    if bbox_poly.intersects(quad_poly):
        intersection = bbox_poly.intersection(quad_poly)
        if not intersection.is_empty:
            intersection_area = intersection.area
            quad_area = quad_poly.area
            
            if quad_area > 0:
                coverage = intersection_area / quad_area
                # Need at least 30% coverage
                if coverage > 0.3:
                    # Also check IoU
                    union_area = bbox_poly.union(quad_poly).area
                    if union_area > 0:
                        iou = intersection_area / union_area
                        if iou >= iou_threshold:
                            return True
    
    return False


def detect_player_tracking(video_path: str, verbose: bool = False) -> Dict[int, Dict[int, np.ndarray]]:
    """Detect and track all players in the video.
    
    Args:
        video_path: Path to video file
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary: {frame_number: {track_id: bbox_xyxy}}
        where bbox_xyxy is [x1, y1, x2, y2] in pixel coordinates
    """
    players = {}
    
    results = player_tracking_model.track(
        video_path,
        stream=True,
        tracker="bytetrack.yaml",
        save=False,
        verbose=verbose
    )
    
    frame_number = 0
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            xyxy = boxes.xyxy.cpu().numpy()
            
            frame_players = {}
            for track_id, bbox in zip(track_ids, xyxy):
                frame_players[track_id] = bbox
            
            players[frame_number] = frame_players
        
        frame_number += 1
    
    return players


def assign_jersey_numbers(
    players: Dict[int, Dict[int, np.ndarray]],
    jersey_quads: Dict,
    video_path: str,
    verbose: bool = False
) -> Dict[int, Optional[str]]:
    """Assign jersey numbers to players by checking intersections with jersey quads.
    
    Args:
        players: Dictionary from detect_player_tracking {frame_number: {track_id: bbox}}
        jersey_quads: Dictionary from JSON file {frame_number: {jersey_key: quad}}
        video_path: Path to video file (to get image dimensions)
    
    Returns:
        Dictionary: {track_id: jersey_number} (jersey_number can be None if not found)
    """
    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Track jersey assignments: track_id -> jersey_number
    # For each track_id, collect all detected jersey numbers with their frame counts
    track_jersey_candidates: Dict[int, Dict[str, int]] = {}  # {track_id: {jersey_number: count}}
    
    # Process each frame - match frame numbers exactly
    frames_processed = 0
    frames_with_matches = 0
    
    for frame_num, frame_players in players.items():
        frame_key = str(frame_num)
        frame_jersey_quads = jersey_quads.get(frame_key, {})
        
        if not frame_jersey_quads:
            continue
        
        frames_processed += 1
        
        # Process each player in this frame
        for track_id, bbox_xyxy in frame_players.items():
            bbox_poly = bbox_to_polygon(bbox_xyxy, img_width, img_height)
            
            # Check intersection with each jersey quad in this frame
            for jersey_key, quad in frame_jersey_quads.items():
                # Extract jersey number from key
                # Keys can be: "15", "15?", "0", "0?", etc.
                # Remove trailing "?" if present, but keep the number
                jersey_number = jersey_key.rstrip('?')
                
                # Skip if empty after stripping
                if not jersey_number:
                    continue
                
                try:
                    quad_poly = quad_to_polygon(quad, img_width, img_height)
                    
                    if check_intersection(bbox_poly, quad_poly):
                        if track_id not in track_jersey_candidates:
                            track_jersey_candidates[track_id] = {}
                        
                        # Count occurrences of each jersey number for this track
                        if jersey_number not in track_jersey_candidates[track_id]:
                            track_jersey_candidates[track_id][jersey_number] = 0
                        track_jersey_candidates[track_id][jersey_number] += 1
                        frames_with_matches += 1
                except Exception as e:
                    # Skip invalid polygons but log for debugging
                    continue
    
    # Assign most common jersey number to each track
    track_to_jersey: Dict[int, Optional[str]] = {}
    for track_id, jersey_counts in track_jersey_candidates.items():
        if jersey_counts:
            # Use most frequently detected jersey number
            # jersey_counts is {jersey_number: count}, find max by count
            most_common_jersey = max(jersey_counts.items(), key=lambda x: x[1])[0]
            track_to_jersey[track_id] = most_common_jersey
    
    if verbose:
        print(f"  Processed {frames_processed} frames with jersey quads")
        print(f"  Found {frames_with_matches} matches between players and jersey quads")
        print(f"  Assigned jersey numbers to {len(track_to_jersey)} tracks")
    
    return track_to_jersey


def assign_teams_by_vertical_line(actions: List[PlayerAction], verbose: bool = False) -> Dict[str, str]:
    """Assign teams to players based on a vertical line that divides them as equally as possible.
    
    Finds the vertical line (strictly vertical, x-coordinate) that best splits players into two teams.
    
    Args:
        actions: List of PlayerAction objects with positions
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary mapping jersey_number to team ("team_A" or "team_B")
    """
    from collections import defaultdict
    
    # Collect positions per player (jersey_number)
    # Use average position across all actions for each player
    player_positions = defaultdict(list)  # {jersey_number: [(x, y), ...]}
    
    for action in actions:
        if action.jersey_number and action.position:
            player_positions[action.jersey_number].append(action.position)
    
    if len(player_positions) < 2:
        # Need at least 2 players to assign teams
        if verbose:
            print("  Not enough players with positions to assign teams")
        return {}
    
    # Compute average x-coordinate for each player
    player_avg_x = {}  # {jersey_number: avg_x}
    for jersey_number, positions in player_positions.items():
        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            player_avg_x[jersey_number] = avg_x
    
    if len(player_avg_x) < 2:
        if verbose:
            print("  Not enough players with valid positions to assign teams")
        return {}
    
    # Find the vertical line that divides players as equally as possible
    # Sort players by x-coordinate
    sorted_players = sorted(player_avg_x.items(), key=lambda x: x[1])
    jersey_numbers = [jn for jn, _ in sorted_players]
    x_coords = [x for _, x in sorted_players]
    
    # Try all possible split points and find the one that divides most equally
    best_split_idx = len(jersey_numbers) // 2
    best_balance = float('inf')
    
    # Try splits at different positions
    for split_idx in range(1, len(jersey_numbers)):
        left_count = split_idx
        right_count = len(jersey_numbers) - split_idx
        balance = abs(left_count - right_count)
        
        if balance < best_balance:
            best_balance = balance
            best_split_idx = split_idx
    
    # Assign teams based on the split
    jersey_to_team = {}
    for i, jersey_number in enumerate(jersey_numbers):
        if i < best_split_idx:
            jersey_to_team[jersey_number] = "team_A"
        else:
            jersey_to_team[jersey_number] = "team_B"
    
    if verbose:
        team_a_players = [jn for jn, team in jersey_to_team.items() if team == "team_A"]
        team_b_players = [jn for jn, team in jersey_to_team.items() if team == "team_B"]
        split_x = x_coords[best_split_idx] if best_split_idx < len(x_coords) else 0.5
        print(f"  Vertical line at x={split_x:.3f} divides players:")
        print(f"    Team A ({len(team_a_players)} players): {sorted(team_a_players)}")
        print(f"    Team B ({len(team_b_players)} players): {sorted(team_b_players)}")
    
    return jersey_to_team


def detect_stats_over_clip(video_path: str, jersey_quads_path: str, start_time: float = 0.0, end_time: float = 0.0, verbose: bool = False) -> StatsOverClip:
    """Detect player stats over a video clip.
    
    Args:
        video_path: Path to video file
        jersey_quads_path: Path to JSON file with jersey quadrilaterals
        start_time: Start time of clip (for StatsOverClip)
        end_time: End time of clip (for StatsOverClip)
        verbose: Whether to print verbose output
    
    Returns:
        StatsOverClip with detected actions
    """
    # Load jersey quads
    with open(jersey_quads_path, 'r') as f:
        jersey_quads = json.load(f)

    # Step 1: Get all players with tracking
    if verbose:
        print("Detecting players with tracking...")
    players = detect_player_tracking(video_path, verbose=verbose)
    if verbose:
        print(f"Found players in {len(players)} frames")
    
    # Step 2: Assign jersey numbers to players
    if verbose:
        print("Assigning jersey numbers...")
        # Debug: Count total jersey quads
        total_jersey_quads = sum(len(quads) for quads in jersey_quads.values())
        total_frames_with_jerseys = len(jersey_quads)
        print(f"  Found {total_frames_with_jerseys} frames with {total_jersey_quads} total jersey quads in JSON")
    track_to_jersey = assign_jersey_numbers(players, jersey_quads, video_path, verbose=verbose)
    
    # Step 3: Detect actions using YOLO action model
    if verbose:
        print("Detecting actions...")
    action_results = action_model.predict(video_path, stream=True, save=False, verbose=verbose)
    
    # Get video dimensions and FPS
    cap = cv2.VideoCapture(video_path)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Get class names from model
    class_names = action_model.names if hasattr(action_model, 'names') else {}
    
    # Collect all detected actions with frame numbers
    # Structure: (action_name, jersey_number, frame_number, confidence, position_x, position_y)
    action_detections = []
    frame_number = 0
    
    for result in action_results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.int().cpu().numpy()
            
            # Get players in this frame
            frame_players = players.get(frame_number, {})
            
            # Process each detected action
            for bbox, confidence, class_id in zip(xyxy, conf, cls):
                action_name = class_names.get(int(class_id), f"class_{class_id}")
                action_bbox_poly = bbox_to_polygon(bbox, img_width, img_height)
                
                # Calculate normalized center position of action bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                # Normalize to [0, 1]
                normalized_x = center_x / img_width
                normalized_y = center_y / img_height
                
                # Find which player this action intersects with
                best_track_id = None
                best_iou = 0.0
                min_iou_threshold = 0.1  # Minimum IoU to consider as intersection
                
                for track_id, player_bbox in frame_players.items():
                    player_bbox_poly = bbox_to_polygon(player_bbox, img_width, img_height)
                    
                    try:
                        if not action_bbox_poly.is_valid or not player_bbox_poly.is_valid:
                            continue
                        
                        intersection = action_bbox_poly.intersection(player_bbox_poly)
                        if intersection.is_empty:
                            continue
                        
                        union = action_bbox_poly.union(player_bbox_poly)
                        iou = intersection.area / union.area if union.area > 0 else 0.0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_track_id = track_id
                    except Exception:
                        continue
                
                # Only process actions that intersect with at least one player
                if best_track_id is None or best_iou < min_iou_threshold:
                    continue
                
                # Assign jersey number or unique ID
                jersey_number = track_to_jersey.get(best_track_id)
                if jersey_number is not None:
                    player_id = str(jersey_number)
                else:
                    player_id = f"?"
                
                # Store detection with frame number and position
                action_detections.append((action_name, player_id, frame_number, float(confidence), normalized_x, normalized_y))
        
        frame_number += 1
    
    if verbose:
        print(f"Detected {len(action_detections)} raw action detections")
    
    # Group actions by (action_name, jersey_number) and compute start/end times
    from collections import defaultdict
    grouped_actions = defaultdict(lambda: {'frames': [], 'confidences': [], 'positions': []})
    
    for action_name, player_id, frame_num, confidence, pos_x, pos_y in action_detections:
        key = (action_name, player_id)
        grouped_actions[key]['frames'].append(frame_num)
        grouped_actions[key]['confidences'].append(confidence)
        grouped_actions[key]['positions'].append((pos_x, pos_y))
    
    # Convert grouped detections to PlayerAction objects
    actions = []
    for (action_name, player_id), data in grouped_actions.items():
        frames = data['frames']
        confidences = data['confidences']
        positions = data['positions']
        
        # Compute start and end times from frame numbers
        min_frame = min(frames)
        max_frame = max(frames)
        action_start_time = start_time + (min_frame / fps) if fps > 0 else start_time
        action_end_time = start_time + (max_frame / fps) if fps > 0 else start_time
        
        # Use average confidence for the group
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Use average position for the group
        if positions:
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            avg_position = (avg_x, avg_y)
        else:
            avg_position = None
        
        action = PlayerAction(
            action=action_name,
            confidence=avg_confidence,
            jersey_number=player_id,
            start_time=action_start_time,
            end_time=action_end_time,
            position=avg_position
        )
        actions.append(action)
    
    if verbose:
        print(f"Grouped into {len(actions)} action groups")
    
    # Filter out consecutive duplicate actions (spikes/attacks and sets)
    # Actions that can't be consecutive for the same player: spike/attack, set
    # Map variations to canonical names
    action_name_mapping = {
        'spike': 'spike',
        'attack': 'spike',  # Treat attack as spike
        'set': 'set'
    }
    
    # Sort actions by start_time for each player
    actions.sort(key=lambda x: (x.jersey_number or "", x.start_time))
    
    # Filter consecutive duplicates
    filtered_actions = []
    i = 0
    while i < len(actions):
        current_action = actions[i]
        
        # Check if this action type should be filtered for consecutive duplicates
        action_name_lower = current_action.action.lower()
        canonical_action = action_name_mapping.get(action_name_lower)
        
        if canonical_action:
            # Look ahead for consecutive actions of the same canonical type for the same player
            j = i + 1
            best_action = current_action
            consecutive_count = 1
            
            while j < len(actions):
                next_action = actions[j]
                next_action_name_lower = next_action.action.lower()
                next_canonical_action = action_name_mapping.get(next_action_name_lower)
                
                # Check if it's the same player and same canonical action type
                if (next_action.jersey_number == current_action.jersey_number and 
                    next_canonical_action == canonical_action):
                    consecutive_count += 1
                    # Keep track of action with highest confidence
                    if next_action.confidence > best_action.confidence:
                        best_action = next_action
                    j += 1
                else:
                    break
            
            # If we found consecutive duplicates, only keep the best one
            if consecutive_count > 1:
                filtered_actions.append(best_action)
                i = j
                continue
        
        # For non-filtered actions or single occurrences, add as-is
        filtered_actions.append(current_action)
        i += 1
    
    actions = filtered_actions
    if verbose:
        print(f"After filtering consecutive duplicates: {len(actions)} action groups")
    
    # Final sort by start_time (chronological order across all players)
    actions.sort(key=lambda x: x.start_time)
    
    # Assign teams based on vertical line division
    if verbose:
        print("Assigning teams based on vertical line division...")
    jersey_to_team = assign_teams_by_vertical_line(actions, verbose=verbose)
    
    # Assign team to each action based on jersey_number
    for action in actions:
        if action.jersey_number and action.jersey_number in jersey_to_team:
            action.team = jersey_to_team[action.jersey_number]
    
    # Filter out serves that are not the first serve temporally
    # Only keep the first serve (earliest in time), remove all others
    serve_actions = [action for action in actions if action.action.lower() in ['serve', 'serving']]
    if serve_actions:
        first_serve = serve_actions[0]  # Already sorted by start_time, so first one is earliest
        first_serve_time = first_serve.start_time
        # Remove all serves except the first one
        actions = [action for action in actions if not (action.action.lower() in ['serve', 'serving'] and action.start_time > first_serve_time)]
        if verbose:
            print(f"After filtering serves (kept first serve at {first_serve_time:.2f}s): {len(actions)} action groups")
    
    return StatsOverClip(
        actions=actions,
        start_time=start_time,
        end_time=end_time
    )

def save_stats_to_json(stats: StatsOverClip, output_path: str):
    """Save StatsOverClip to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)

def process_video(video_path: str, jersey_quads_path: str, output_path: str, verbose: bool = False) -> StatsOverClip:
    """Process a single video and save results.
    
    Args:
        video_path: Path to video file
        jersey_quads_path: Path to JSON file with jersey quadrilaterals
        output_path: Path to save results JSON
        verbose: Whether to print verbose output
    
    Returns:
        StatsOverClip with detected actions
    """
    import re
    
    # Try to extract start and end time from filename
    # Pattern: name_start_end_*.mp4
    filename = os.path.basename(video_path)
    match = re.search(r'(\d+\.\d+)_(\d+\.\d+)_', filename)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
    else:
        start_time = 0.0
        end_time = 0.0
    
    if verbose:
        print(f"\nProcessing: {os.path.basename(video_path)}")
    
    result = detect_stats_over_clip(video_path, jersey_quads_path, start_time, end_time, verbose=verbose)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_stats_to_json(result, output_path)
    
    if verbose:
        print(f"\nStats Summary:")
        print(f"  Total action groups: {len(result.actions)}")
        print(f"  Clip time range: {result.start_time:.2f}s - {result.end_time:.2f}s")
        for i, action in enumerate(result.actions[:10]):  # Print first 10
            jersey_str = f" (Jersey: {action.jersey_number})" if action.jersey_number else ""
            print(f"  {i+1}. {action.action}: {action.confidence:.3f}{jersey_str} [{action.start_time:.2f}s - {action.end_time:.2f}s]")
        if len(result.actions) > 10:
            print(f"  ... and {len(result.actions) - 10} more actions")
        print(f"  Saved results to: {output_path}")
    
    return result

if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Detect player stats from video clips')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--jersey-quads', type=str, help='Path to jersey quadrilaterals JSON file')
    parser.add_argument('--input-dir', type=str, help='Directory containing video files (will process all .mp4 files)')
    parser.add_argument('--jersey-quads-dir', type=str, default='output_quadrilaterals_with_jerseys',
                       help='Directory containing jersey quad JSON files (default: output_quadrilaterals_with_jerseys)')
    parser.add_argument('--output-dir', type=str, default='output_stats',
                       help='Directory to save results JSON files (default: output_stats)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output and YOLO verbosity')
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Process all videos in directory
        input_dir = Path(args.input_dir)
        jersey_quads_dir = Path(args.jersey_quads_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = list(input_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"No video files found in {input_dir}")
            exit(1)
        
        print(f"Found {len(video_files)} video files to process")
        
        # Filter videos that have corresponding jersey quad files
        valid_videos = []
        for video_path in video_files:
            video_name = video_path.stem
            jersey_quads_path = jersey_quads_dir / f"quadrilaterals_with_jerseys_{video_name}.json"
            if jersey_quads_path.exists():
                valid_videos.append((video_path, jersey_quads_path))
            elif args.verbose:
                print(f"Skipping {video_name}: jersey quads file not found")
        
        processed = 0
        failed = 0
        
        # Use tqdm progress bar if not verbose
        iterator = tqdm(valid_videos, desc="Processing videos", unit="video") if not args.verbose else valid_videos
        
        for video_path, jersey_quads_path in iterator:
            video_name = video_path.stem
            output_path = output_dir / f"stats_{video_name}.json"
            
            try:
                process_video(str(video_path), str(jersey_quads_path), str(output_path), verbose=args.verbose)
                processed += 1
            except Exception as e:
                failed += 1
                error_msg = f"Error processing {video_name}: {e}"
                if args.verbose:
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                else:
                    # Update progress bar description with error
                    if hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix_str(f"Error: {video_name}")
        
        print(f"\nProcessed {processed}/{len(valid_videos)} videos successfully")
        if failed > 0:
            print(f"Failed: {failed} videos")
        print(f"Results saved to: {output_dir}")
    
    elif args.video and args.jersey_quads:
        # Process single video
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(args.video).stem
        output_path = output_dir / f"stats_{video_name}.json"
        
        process_video(args.video, args.jersey_quads, str(output_path), verbose=args.verbose)
    
    else:
        parser.print_help()
        exit(1)
