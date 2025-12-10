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

    def search(self, query: str, namespace: str = "test-namespace", top_k: int = 50, video_id: Optional[str] = None):
        query_dict = {
            "top_k": top_k,
            "inputs": {
                'text': query
            }
        }
        
        results = self.index.search(
            namespace=namespace,
            query=query_dict
        )

        return [result for result in results['result']['hits']]

    def search_vector(self, vector: list[float], namespace: str = "test-namespace", top_k: int = 50, ids: list[str] = 'all' if config.EMBED_FOR_ME else [1]):
        # Pinecone expects vector to be wrapped in a dictionary with "values" key
        query_dict = {
            "top_k": top_k,
            "vector": {
                "values": vector
            }
        }
        
        # Add filter if video_id is provided and not "all"
        if ids and ids != "all":
            query_dict["filter"] = {"video_id": {"$in": ids}}
        
        results = self.index.search(
            namespace=namespace,
            query=query_dict
        )
        return [result for result in results['result']['hits']]

if __name__ == "__main__":
    pinecone_client = PineconeClient(config.PINECONE_INDEX_NAME)
    results = pinecone_client.search("Block", namespace="test-namespace", top_k=10)
    print(results)