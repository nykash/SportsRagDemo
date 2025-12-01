from pinecone import Pinecone
from typing import Optional, Any
import os
from config import get_config

config = get_config()

class PineconeClient:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(self.index_name)

    def initialize_index(self) -> bool:
        if not self.pc.has_index(self.index_name):
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "summary"}
                }
            )
            return True
        return False

    def upsert(self, dicts_to_upsert: list[dict[str, Any]], namespace: str = "test-namespace"):
        # batch size is 90
        batch_size = 90
        for i in range(0, len(dicts_to_upsert), batch_size):
            self.index.upsert_records(namespace=namespace, records=dicts_to_upsert[i:i+batch_size])

    def delete_index(self, namespace: str = "test-namespace"):
        return self.index.delete(namespace=namespace, delete_all=True)

    def search(self, query: str, namespace: str = "test-namespace", top_k: int = 5):
        results = self.index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    'text': query
                }
            }
        )

        return [result for result in results['result']['hits']]