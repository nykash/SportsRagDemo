from moviepy.editor import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess

output_dir = "data/splices"
os.makedirs(output_dir, exist_ok=True)

video_splices = {
    "1.mp4": "1_timestamps.json",
    "4.mp4": "4_timestamps.json"
    "2.mp4": "2_timestamps.json",
    "3.mp4": "3_timestamps.json",
    "5.mp4": "5_timestamps.json"
}

def process_clip(args):
    """Process a single video clip."""
    video_full_path, start_time, end_time, out_path, timestamp_value = args
    clip = VideoFileClip(video_full_path).subclip(start_time, end_time)
    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    clip.close()
    return out_path

# Collect all tasks
all_tasks = []
for video_path, timestamps_path in video_splices.items():
    timestamps = json.load(open(timestamps_path))
    video_full_path = os.path.join("data", video_path)
    for timestamp, next_timestamp in zip(timestamps[:-1], timestamps[1:]):
        start_time = timestamp[0]
        end_time = next_timestamp[0]
        out_path = f"{output_dir}/{video_path.split('.')[0]}_{start_time}_{end_time}_{timestamp[1]}.mp4"
        all_tasks.append((video_full_path, start_time, end_time, out_path, timestamp[1]))

# Process clips in parallel using threads (better for MoviePy I/O operations)
max_workers = min(os.cpu_count() or 4, len(all_tasks))
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_clip, task): task for task in all_tasks}
    for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Processing clips"):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing clip: {e}")