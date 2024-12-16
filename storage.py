from abc import ABC, abstractmethod
import os
from typing import Optional

class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload a file and return its public URL"""
        pass
    
    @abstractmethod
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        """Get a signed/public URL for a file"""
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Delete a file"""
        pass

class GCSProvider(StorageProvider):
    """Google Cloud Storage implementation"""
    
    def __init__(self, bucket_name: str):
        from google.cloud import storage
        from google.cloud.exceptions import GoogleCloudError
        
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        return self.get_url(remote_path)
        
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        blob = self.bucket.blob(remote_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=expires_in,
            method="GET"
        )
        return url
        
    def delete_file(self, remote_path: str) -> bool:
        blob = self.bucket.blob(remote_path)
        blob.delete()
        return True

class S3Provider(StorageProvider):
    """AWS S3 implementation"""
    
    def __init__(self, bucket_name: str, region_name: Optional[str] = None):
        import boto3
        from botocore.exceptions import ClientError
        
        self.bucket_name = bucket_name
        self.client = boto3.client('s3', region_name=region_name)
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        self.client.upload_file(local_path, self.bucket_name, remote_path)
        return self.get_url(remote_path)
        
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        url = self.client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': self.bucket_name,
                'Key': remote_path
            },
            ExpiresIn=expires_in
        )
        return url
        
    def delete_file(self, remote_path: str) -> bool:
        self.client.delete_object(Bucket=self.bucket_name, Key=remote_path)
        return True

class LocalProvider(StorageProvider):
    """Local filesystem implementation for development/testing"""
    
    def __init__(self, base_path: str, base_url: str):
        self.base_path = base_path
        self.base_url = base_url
        os.makedirs(base_path, exist_ok=True)
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        target_path = os.path.join(self.base_path, remote_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(local_path, 'rb') as src, open(target_path, 'wb') as dst:
            dst.write(src.read())
        return self.get_url(remote_path)
        
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        return f"{self.base_url}/{remote_path}"
        
    def delete_file(self, remote_path: str) -> bool:
        target_path = os.path.join(self.base_path, remote_path)
        if os.path.exists(target_path):
            os.remove(target_path)
            return True
        return False

def get_storage_provider(provider_type: str = None) -> StorageProvider:
    """Factory function to get the configured storage provider"""
    provider_type = provider_type or os.getenv('STORAGE_PROVIDER', 'local')
    
    if provider_type == 'gcs':
        bucket_name = os.getenv('GCS_BUCKET')
        if not bucket_name:
            raise ValueError("GCS_BUCKET environment variable is required for GCS storage")
        return GCSProvider(bucket_name)
        
    elif provider_type == 's3':
        bucket_name = os.getenv('S3_BUCKET')
        region = os.getenv('AWS_REGION')
        if not bucket_name:
            raise ValueError("S3_BUCKET environment variable is required for S3 storage")
        return S3Provider(bucket_name, region)
        
    elif provider_type == 'local':
        base_path = os.getenv('LOCAL_STORAGE_PATH', './storage')
        base_url = os.getenv('LOCAL_STORAGE_URL', 'http://localhost:5000/storage')
        return LocalProvider(base_path, base_url)
        
    else:
        raise ValueError(f"Unsupported storage provider: {provider_type}") 