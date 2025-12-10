import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

def get_config():
    return Settings()