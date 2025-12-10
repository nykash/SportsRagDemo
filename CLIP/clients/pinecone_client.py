from pinecone import Pinecone, ServerlessSpec
from typing import Optional, Any, List, Union
import time
from config import get_config

config = get_config()

class PineconeClient:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        # We REMOVED self.index = self.pc.Index(...) from here
        # so it doesn't crash if the index is missing.
        self._index = None

    @property
    def index(self):
        """Lazy loads the index connection only when needed."""
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def index_exists(self) -> bool:
        """Checks if the index currently exists."""
        return self.index_name in self.pc.list_indexes().names()

    def create_vector_index(self, dimension: int, metric: str = "cosine"):
        """
        Creates a standard serverless index for raw vectors (e.g., CLIP embeddings).
        """
        if not self.index_exists():
            print(f"Creating vector index '{self.index_name}' (dim={dimension})...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index is ready.")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def upsert_vectors(self, vectors: List[tuple], batch_size: int = 100, namespace: str = ""):
        """
        Upserts raw vectors.
        Expected format: [(id, vector_list, metadata_dict), ...]
        """
        print(f"Upserting {len(vectors)} vectors...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

    def search_by_vector(self, vector: List[float], top_k: int = 5, namespace: str = ""):
        """
        Search using a raw vector.
        """
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )