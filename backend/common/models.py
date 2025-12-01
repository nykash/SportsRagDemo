from pydantic import BaseModel
import uuid
from pydantic import Field

class Transcription(BaseModel):
    start_time: float
    end_time: float
    text: str
    video_id: str

class TranscriptionWithSummary(Transcription):
    summary: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class TranscriptionRetrievalResult(BaseModel):
    id: str
    summary: str
    start_time: float
    end_time: float
    text: str
    score: float
    video_id: str