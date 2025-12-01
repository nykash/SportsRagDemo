from clients.pinecone_client import PineconeClient
from config import get_config
from common.models import TranscriptionRetrievalResult
from typing import List

config = get_config()

def retrieve(query: str, namespace: str = "test-namespace") -> List[TranscriptionRetrievalResult]:
    pinecone_client = PineconeClient(index_name=config.PINECONE_INDEX_NAME)
    search_results = pinecone_client.search(query, namespace=namespace)

    return [TranscriptionRetrievalResult(
            id=result["_id"],
            summary=result["fields"]["summary"],
            start_time=result["fields"]["start_time"],
            end_time=result["fields"]["end_time"],
            text=result["fields"]["text"],
            score=result["_score"],
            video_id=result["fields"]["video_id"]
        ) for result in search_results]