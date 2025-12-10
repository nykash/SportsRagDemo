from flask import request, jsonify
from config import get_config
from utils.grab_s3_uri import s3_key_from_segment
from client.pinecone_client import PineconeClient
from client.s3_client import S3Client
import json
import os
import shutil
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.text_encoder import encode_text


highlights = json.load(open('highlights.json'))

config = get_config()
s3_client = S3Client()

# Directory for temporary video files (must match app.py)
TEMP_VIDEO_DIR = 'temp_videos'

def cleanup_temp_videos():
    """Delete all files in the temp_videos directory"""
    if os.path.exists(TEMP_VIDEO_DIR):
        try:
            shutil.rmtree(TEMP_VIDEO_DIR)
            os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
            print(f"Cleaned up {TEMP_VIDEO_DIR} directory")
        except Exception as e:
            print(f"Error cleaning up temp videos: {e}")

def download_video_from_s3(s3_key: str) -> tuple:
    """Download video from S3 to local temp directory and return (s3_key, local_url) tuple"""
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
    
    # Create a safe filename from S3 key
    # Replace slashes and special characters
    safe_filename = s3_key.replace('/', '_').replace('\\', '_')
    local_path = os.path.join(TEMP_VIDEO_DIR, safe_filename)
    
    # Download if not already exists
    if not os.path.exists(local_path):
        try:
            s3_client.download_file(s3_key, local_path)
            print(f"Downloaded {s3_key} to {local_path}")
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return (s3_key, None)
    
    # Return the URL path (relative to Flask static serving)
    return (s3_key, f"/{TEMP_VIDEO_DIR}/{safe_filename}")


@dataclass
class Event:
    event_type: str
    start: float
    end: float
    team: str
    player: str

def simple_action_filter(events: dict, action: str):
    """Check if event contains the specified action"""
    for event in events:
        event_type = event.get('action', '') or event.get('event_type', '')
        if action.lower() in str(event_type).lower():
            return True
    return False

def long_rally_filter(events: list):
    """Check if rally has 3 or more sets"""
    num_set = 0
    for event in events:
        event_type = event.get('action', '') or event.get('event_type', '')
        if 'set' in str(event_type).lower():
            num_set += 1
        if num_set >= 3:
            return True
    return False

def successful_defense_filter(events: list):
    """Check if defense is followed by spike/set/block"""
    found_spike = False
    ended_spike = -1
    for i, event in enumerate(events):
        event_type = event.get('action', '') or event.get('event_type', '')
        if 'spike' in str(event_type).lower():
            found_defense = True
            ended = i
            break
    
    found_defense = False
    ended = -1
    for i, event in enumerate(events[ended_spike:]):
        event_type = event.get('action', '') or event.get('event_type', '')
        if 'defense' in str(event_type).lower():
            found_defense = True
            ended = i
            break
    
    if not found_defense:
        return False
    
    spiked = False
    setted = False
    blocked = False
    for event in events[ended:]:
        event_type = event.get('action', '') or event.get('event_type', '')
        event_str = str(event_type).lower()
        if 'spike' in event_str:
            spiked = True
        elif 'set' in event_str:
            setted = True
        elif 'block' in event_str:
            blocked = True

    return spiked and setted and blocked

def filter_results_by_tags(results: list, tags: list):
    """Filter results based on selected tags"""
    if not tags or len(tags) == 0:
        return results
    
    filtered_results = []
    for result in results:
        # Get events from the result
        events = result.get('events', {})
        
        # Check if result matches any of the tags
        matches_all = True
        for tag in tags:
            tag = tag.lower()
            
            if tag in ['block', 'serve', 'set', 'defense', 'spike'] and not simple_action_filter(events, tag):
                matches_all = False
                break
            elif tag == 'long_rally' and not long_rally_filter(events):
                matches_all = False
                break
            elif tag == 'successful_defense' and not successful_defense_filter(events):
                matches_all = False
                break
        
        if matches_all:
            filtered_results.append(result)
    
    return filtered_results

def closest_commentary(video_id: str, time_start: float, time_stop: float, tags: list = None, left_clip_margins: float = 1, right_clip_margins: float = 0):
    first_in_range = None
    first_in_range_index = None
    for i, highlight in enumerate(highlights[video_id]):
        if highlight['time_start'] >= time_start:
            first_in_range = highlight
            first_in_range_index = i
            break
    if first_in_range is None:
        return None
    
    results = []
    for i in range(first_in_range_index - left_clip_margins, first_in_range_index + right_clip_margins + 1):
        if i < 0 or i >= len(highlights[video_id]):
            continue
        s3_key = s3_key_from_segment(video_id, highlights[video_id][i]['time_start'], highlights[video_id][i]['time_stop'])
        results.append({
            "s3_key": s3_key,
            "video_id": video_id,
            "time_start": highlights[video_id][i]['time_start'],
            "time_stop": highlights[video_id][i]['time_stop'],
            "score": 1,
            "events": highlights[video_id][i]["actions"]
        })
    
    results = filter_results_by_tags(results, tags)

    return results

def handle_retrieve():
    """Handle POST request with question and tags"""
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        return jsonify({}), 200
    
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    print(data)
    question = data.get('question') if data else None
    tags = data.get('tags', []) if data else []
    video_id = data.get('video_id') if data else None

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # Clean up old videos before downloading new ones
    cleanup_temp_videos()

    # Use tags as namespace if provided, otherwise use default
    namespace = "test-namespace" if config.EMBED_FOR_ME else "__default__"
    
    retrieve_client = PineconeClient(config.PINECONE_INDEX_NAME)
    if config.EMBED_FOR_ME:
        results = retrieve_client.search(question, namespace=namespace, top_k=20)
    else:
        results = retrieve_client.search_vector(encode_text(question), namespace=namespace, top_k=20, ids='all')

    print(results)

    retrieved_clips = []

    for result in results:
        video_id = result.get('fields', {}).get('video_id')
        video_id = str(int(video_id))
        if "long_video" in video_id:
            video_id = "1"
        time_start = result.get('fields', {}).get('start_time')
        time_stop = result.get('fields', {}).get('end_time')

        new_clips = closest_commentary(video_id, time_start, time_stop, tags=tags)
        if new_clips is not None:
            for clip in new_clips:
                clip['score'] = result.get('_score', 0)
            retrieved_clips.extend(new_clips)
    
    # Collect all S3 keys that need to be downloaded
    s3_keys_to_download = []
    for clip in retrieved_clips:
        s3_key = clip.get('s3_key')
        if s3_key:
            s3_keys_to_download.append((clip, s3_key))
    
    # Download videos in parallel
    print(f"Downloading {len(s3_keys_to_download)} videos in parallel...")
    max_workers = min(10, len(s3_keys_to_download)+1)  # Limit to 10 concurrent downloads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_clip = {
            executor.submit(download_video_from_s3, s3_key): (clip, s3_key)
            for clip, s3_key in s3_keys_to_download
        }
        
        # Process completed downloads
        for future in as_completed(future_to_clip):
            clip, s3_key = future_to_clip[future]
            try:
                downloaded_s3_key, local_url = future.result()
                if local_url:
                    clip['url'] = local_url
                # Remove s3_key from response, we only need url
                clip.pop('s3_key', None)
            except Exception as e:
                print(f"Error processing download for {s3_key}: {e}")
                # Remove s3_key even if download failed
                clip.pop('s3_key', None)
    
    print(f"Completed downloading {len(s3_keys_to_download)} videos")
    
    return jsonify({
        'question': question,
        'tags': tags,
        'results': retrieved_clips,
        'count': len(retrieved_clips)
    }), 200
