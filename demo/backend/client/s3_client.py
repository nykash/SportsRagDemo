import boto3
import os
from config import get_config

config = get_config()

class S3Client:
    def __init__(self, bucket_name: str = config.AWS_BUCKET_NAME):
        self.s3 = boto3.client('s3', aws_access_key_id=config.AWS_ACCESS_KEY_ID, aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY, region_name=config.AWS_REGION)
        self.bucket_name = bucket_name

    def read_file(self, key: str):
        return self.s3.get_object(Bucket=self.bucket_name, Key=key)['Body'].read()

    def download_file(self, key: str, local_path: str):
        """Download a file from S3 to local path"""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, key, local_path)
        return local_path

    def generate_presigned_url(self, key: str, expires_in: int = 3600):
        """
        Generate a presigned URL for S3 object access.
        Increased expiration time to 1 hour for better video playback.
        """
        return self.s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": key,
                "ResponseContentType": "video/mp4"  # Explicitly set content type
            },
            ExpiresIn=expires_in
        )

    


    
    