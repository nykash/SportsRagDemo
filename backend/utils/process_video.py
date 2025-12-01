from clients.whisper_client import WhisperClient
from clients.ollama_client import OllamaClient
from clients.pinecone_client import PineconeClient
from common.models import Transcription, TranscriptionWithSummary
from typing import List, Optional
from config import get_config
from pathlib import Path

config = get_config()

def transcribe_video(video_path: str) -> List[Transcription]:
    whisper_client = WhisperClient()

    return whisper_client.transcribe_and_align(video_path)


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

def encode_text(ollama_client: OllamaClient, text: str, previous_commentary: str) -> str:
    return ollama_client.generate_single_turn(
        context="You are a helpful assistant that listens to Volleyball commentary and summarizes the events that happened in the video. The previous commentary is: " + previous_commentary + ".",
        instruction=Path("prompt_templates/encode.txt").read_text().replace(
            "{{Insert text from your CSV here}}",
            text.strip()
        )
    )

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
