import os
from dotenv import load_dotenv

load_dotenv()

def get_config():
    return Config()

class Config:
    EMBED_FOR_ME = False

    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME') if EMBED_FOR_ME else "sports-rag-clip"

    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
