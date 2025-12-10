import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline
from common.models import Transcription
from typing import List, Optional
from transformers import pipeline  # Lowercase 'p' (Factory function)

"""
model_size: 'tiny', 'base', 'small', 'medium', 'large-v2'
device: 'cpu' or 'cuda'
compute_type: 'int8' (lower mem) or 'float16' (faster on GPU)
hf_token: HuggingFace token (Required for Diarization/PyAnnote)
"""
class WhisperClient:
    def __init__(self, model_size="small", device="cpu", compute_type="int8", hf_token=None):
        self.device = device
        self.compute_type = compute_type
        self.batch_size = 16
        self.hf_token = hf_token
        
        print(f"Whisper Model: {model_size} on {self.device}...")
        self.model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)

    def transcribe_and_align(self, audio_file: str) -> List[Transcription]:
        print(f"Loading audio: {audio_file}")
        audio = whisperx.load_audio(audio_file)
        
        print("Transcribing...")
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        
        print("Aligning...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=self.device
        )
        
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        del model_a
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Diarization (Identify Speakers)
        if self.hf_token:
            print("Diarizing (Identifying speakers)...")
            diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
            diarize_segments = diarize_model(audio)
            
            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Cleanup Diarization Model
            del diarize_model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        else:
            print("Skipping Diarization (No HF Token provided)")

        transcriptions = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            
            # Create Transcription object
            formatted_text = f"[{speaker}]: {segment['text'].strip()}"
            
            transcriptions.append(
                Transcription(
                    start_time=segment["start"], 
                    end_time=segment["end"], 
                    text=formatted_text,
                    video_id=audio_file
                )
            )
            
        return transcriptions