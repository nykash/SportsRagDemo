import os   
from dotenv import load_dotenv

load_dotenv()

class Config:
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

configs = {
    "LOCAL": Config(),
    "PRODUCTION": Config(),
}

def get_config():
    return configs[os.getenv("DEV_ENVIRONMENT", "LOCAL")]