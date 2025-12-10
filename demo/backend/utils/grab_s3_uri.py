def s3_key_from_segment(video_id: str, time_start: float, time_stop: float):
    return f"{video_id}/{time_start}_{time_stop}.mp4"