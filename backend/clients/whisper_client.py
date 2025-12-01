import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
from common.models import Transcription
from typing import List

device = "cpu"
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

class WhisperClient:
    def __init__(self):
        self.model = whisperx.load_model("small", device, compute_type=compute_type)

    def transcribe_and_align(self, audio_file: str) -> List[Transcription]:
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=batch_size)
    
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        return [Transcription(start_time=segment["start"], end_time=segment["end"], text=segment["text"], video_id=audio_file) for segment in result["segments"]]
    