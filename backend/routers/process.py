from fastapi import APIRouter, HTTPException
from utils.process_video import transcribe_video, chunk_transcription, summarize_transcription
from common.models import Transcription, TranscriptionWithSummary
from typing import List

router = APIRouter(
    prefix="/process",
    tags=["Process"]
)

@router.post("/video")
async def process_video(video_path: str) -> List[TranscriptionWithSummary]:
    transcription = transcribe_video(video_path)
    chunks = chunk_transcription(transcription)
    summaries = summarize_transcription(chunks)
    
    return summaries