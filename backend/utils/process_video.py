from clients.whisper_client import WhisperClient
from clients.pinecone_client import PineconeClient
from common.models import Transcription, TranscriptionWithSummary
from typing import List, Optional
from config import get_config
from pathlib import Path
import os
import subprocess
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

config = get_config()

# In backend/utils/process_video.py

def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-y", 
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        audio_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_video(video_path):
    # Setup Paths
    clean_video_path = os.path.abspath(os.path.expanduser(video_path))
    
    # Audio temp file stays near the video
    base_name = os.path.splitext(os.path.basename(clean_video_path))[0]
    clean_audio_path = f"{base_name}_temp.mp3"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    experiments_dir = os.path.join(project_root, "experiments")
    
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Final CSV path
    csv_path = os.path.join(experiments_dir, f"{base_name}.csv")

    try:
        print(f"Processing: {clean_video_path}")
        extract_audio(clean_video_path, clean_audio_path)
        client = WhisperClient(model_size="small", device="cpu", compute_type="int8")
        results = client.transcribe_and_align(clean_audio_path)
    
        if results:
            return save_to_csv(results, csv_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None instead of empty list for clearer failure handling
    finally:
        if os.path.exists(clean_audio_path):
            os.remove(clean_audio_path)
            
def save_transcript_to_file(transcriptions, filename="transcript.txt"):
    with open(filename, "w") as f:
        for t in transcriptions:
            minutes = int(t.start_time // 60)
            seconds = int(t.start_time % 60)
            timestamp = f"{minutes:02}:{seconds:02}"
            f.write(f"[{timestamp}] {t.text}\n")
    print(f"Transcript saved to {filename}")

def save_to_csv(transcriptions, output_path):
    headers = ["start_time", "end_time", "speaker", "text"]
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for t in transcriptions:
            # Logic to parse speaker if hidden in text string
            speaker = "Unknown"
            clean_text = t.text
            if "]: " in t.text:
                parts = t.text.split("]: ", 1)
                speaker = parts[0].replace("[", "").strip()
                clean_text = parts[1].strip()

            writer.writerow([
                f"{t.start_time:.2f}", 
                f"{t.end_time:.2f}", 
                speaker, 
                clean_text
            ])
    return output_path

def group_transcriptions_csv(input_csv_path, window_size=7, step_size=2):
    
    if not os.path.exists(input_csv_path):
        print(f"Error: File not found at {input_csv_path}")
        return

    # Read the CSV
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    grouped_data = []

    # Iterate through the dataframe with the defined step size
    for i in range(0, len(df), step_size):
        # Slice the window
        window = df.iloc[i : i + window_size]
        
        if window.empty:
            break

        # Calculate Group Metadata
        start_time = window.iloc[0]['start_time']
        end_time = window.iloc[-1]['end_time']
        duration = end_time - start_time
        
        # Combine Text
        text_content = " ".join(window['text'].astype(str).tolist())
        
        speakers = []
        if 'speaker' in window.columns:
            speakers = window['speaker'].unique().tolist()
        
        grouped_data.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": round(duration, 2),
            "text": text_content,
            "speakers": ", ".join(speakers)
        })

    # Create new DataFrame
    grouped_df = pd.DataFrame(grouped_data)
    
    # Generate Output Filename
    base, ext = os.path.splitext(input_csv_path)
    output_csv_path = f"{base}_grouped{ext}"
    
    # Save to CSV
    grouped_df.to_csv(output_csv_path, index=False)
    print(f"Grouped transcription saved to: {output_csv_path}")
    
    return output_csv_path



def create_video_clips(csv_path, video_path, output_folder_name="clips", max_workers=4):
    clean_csv_path = os.path.abspath(os.path.expanduser(csv_path))
    clean_video_path = os.path.abspath(os.path.expanduser(video_path))
    
    if not os.path.exists(clean_csv_path):
        print(f"Error: CSV not found at {clean_csv_path}")
        return
    if not os.path.exists(clean_video_path):
        print(f"Error: Video not found at {clean_video_path}")
        return

    # Output Directory
    base_dir = os.path.dirname(clean_csv_path)
    clips_dir = os.path.join(base_dir, output_folder_name)
    os.makedirs(clips_dir, exist_ok=True)
    
    print(f"Loading CSV: {clean_csv_path}")
    df = pd.read_csv(clean_csv_path)
    print(f"Generating {len(df)} clips with {max_workers} workers...")

    def process_clip(row_data):
        index, row = row_data
        start_time = row['start_time']
        end_time = row['end_time']
        duration = end_time - start_time
        
        filename = f"clip_{index:03d}_{start_time:.2f}-{end_time:.2f}.mp4"
        output_filepath = os.path.join(clips_dir, filename)
        
        if os.path.exists(output_filepath):
            return f"Skipped {index} (Exists)"

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", clean_video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast", 
            "-crf", "23", 
            "-c:a", "aac",
            "-threads", "1",
            "-loglevel", "error",
            output_filepath
        ]
        
        try:
            subprocess.run(command, check=True)
            return f"Clip {index} done"
        except subprocess.CalledProcessError:
            return f"Clip {index} FAILED"

    tasks = list(df.iterrows())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_clip, tasks))
        
    print(f"✅ Finished! All clips saved in: {clips_dir}")


def chunk_transcription(transcription: List[Transcription], segment_duration: float = 20) -> List[Transcription]:
    """
    Chunk the transcription into segments of segment_duration seconds.
    """
    chunks = []
    last_start_time = 0
    for i in range(len(transcription)):
        if i == 0:
            chunks.append(transcription[i])
            last_start_time = transcription[i].start_time
        elif transcription[i].start_time - last_start_time > segment_duration:
            chunks.append(transcription[i])
            last_start_time = transcription[i].start_time
        else:
            chunks[-1].text += " " + transcription[i].text
            chunks[-1].end_time = transcription[i].end_time
    return chunks

# def encode_text(ollama_client: OllamaClient, text: str, previous_commentary: str) -> str:
#     return ollama_client.generate_single_turn(
#         context="You are a helpful assistant that listens to Volleyball commentary and summarizes the events that happened in the video. The previous commentary is: " + previous_commentary + ".",
#         instruction=Path("prompt_templates/encode.txt").read_text().replace(
#             "{{Insert text from your CSV here}}",
#             text.strip()
#         )
#     )

def summarize_chunk(chunk: Transcription, previous_chunk: Optional[Transcription] = None) -> TranscriptionWithSummary:
    """
    Summarize a single chunk of transcription.
    """
    ollama_client = OllamaClient()
    summary = encode_text(ollama_client, chunk.text, previous_chunk.text if previous_chunk else "")
    return TranscriptionWithSummary(start_time=chunk.start_time, end_time=chunk.end_time, text=chunk.text, summary=summary, video_id=chunk.video_id)

def summarize_transcription(transcription: List[Transcription]) -> List[TranscriptionWithSummary]:
    """
    Summarize the transcription into a list of TranscriptionWithSummary.
    """
    return [summarize_chunk(chunk, transcription[i-1] if i > 0 else Transcription(start_time=0, end_time=0, text="The game starts.", video_id=transcription[0].video_id)) for i, chunk in enumerate(transcription)]

def rag_summarized_transcriptions(summarized_transcriptions: List[TranscriptionWithSummary], namespace: str = "test-namespace") -> None:
    pinecone_client = PineconeClient(index_name=config.PINECONE_INDEX_NAME)
    pinecone_client.upsert([summary.model_dump() for summary in summarized_transcriptions], namespace=namespace)
