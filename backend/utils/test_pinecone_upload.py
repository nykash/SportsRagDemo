import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from clients.pinecone_client import PineconeClient

INDEX_NAME = "sports-rag-clip"
TARGET_MATCH_ID = 6

def main():
    print(f"Connecting to index '{INDEX_NAME}'...")
    client = PineconeClient(index_name=INDEX_NAME)

    # Check stats before deletion
    stats_before = client.index.describe_index_stats()
    print(f"Total vectors BEFORE: {stats_before['total_vector_count']}")

    print(f"Deleting all vectors with match_id = {TARGET_MATCH_ID}...")
    
    # We use the underlying index object to call delete with a filter
    try:
        client.index.delete(
            filter={
                "video_id": {"$eq": TARGET_MATCH_ID}
            }
        )
        print("Deletion command sent successfully.")
    except Exception as e:
        print(f" Error during deletion: {e}")
        return

    import time
    time.sleep(2) 
    stats_after = client.index.describe_index_stats()
    print(f"Total vectors AFTER: {stats_after['total_vector_count']}")

if __name__ == "__main__":
    main()