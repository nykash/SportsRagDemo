from clients.pinecone_client import PineconeClient

client = PineconeClient(index_name="sports-rag-clip")
stats = client.index.describe_index_stats()
print(f"Total Vectors: {stats['total_vector_count']}")
