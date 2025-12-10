# file_system.py
import os
from pathlib import Path
import boto3
from config import get_config

config = get_config()

s3_client = boto3.client("s3") if config.WRITE_TO_S3 else None

if not config.WRITE_TO_S3:
    Path(config.LOCAL_STORAGE_NAME).mkdir(parents=True, exist_ok=True)

def write_file(file_name: str, content: bytes) -> str:
    """
    Save a file either to S3 or local storage.
    
    Args:
        file_name (str): Name of the file.
        content (bytes): File content.
    
    Returns:
        str: Full path or S3 URL of the saved file.
    """
    if config.WRITE_TO_S3:
        s3_client.put_object(Bucket=config.S3_BUCKET_NAME, Key=file_name, Body=content)
        return f"s3://{config.S3_BUCKET_NAME}/{file_name}"
    else:
        file_path = Path(config.LOCAL_STORAGE_NAME) / file_name
        with open(file_path, "wb") as f:
            f.write(content)
        return str(file_path)

def read_file(file_name: str) -> bytes:
    """
    Read a file either from S3 or local storage.
    
    Args:
        file_name (str): Name of the file to read.
    
    Returns:
        bytes: File content.
    """
    if config.WRITE_TO_S3:
        response = s3_client.get_object(Bucket=config.S3_BUCKET_NAME, Key=file_name)
        return response['Body'].read()
    else:
        file_path = Path(config.LOCAL_STORAGE_NAME) / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return f.read()
