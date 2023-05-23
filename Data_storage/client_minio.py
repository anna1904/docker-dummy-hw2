from minio import Minio
from pathlib import Path
import unittest

ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
ENDPOINT = "0.0.0.0:9000"


class MinioClient:
    def __init__(self):
        self.minioClient: Minio = Minio(ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

    def upload_object(self, bucket_name: str, object_name: str, file_path: Path) -> None:
        bucket = self.minioClient.bucket_exists(bucket_name)
        if bucket:
            self.minioClient.fput_object(bucket_name, object_name, file_path)
            print("File has been successfully saved in {}".format(bucket_name))

    def get_object(self, bucket_name: str, object_name: str, file_path: Path):
        bucket = self.minioClient.bucket_exists(bucket_name)
        if bucket:
            result = self.minioClient.fget_object(bucket_name, object_name, file_path)
            print("res:", result)
            print(f"File {result.object_name} has been successfully get from {bucket_name}")
            return result

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        bucket = self.minioClient.bucket_exists(bucket_name)
        if bucket:
            self.minioClient.remove_object(bucket_name, object_name)
            print("File has been successfully deleted from {}".format(bucket_name))

    # As objects are immutable in S3, I first delete the object and upload the new version with the same name
    def update_object(self, bucket_name: str, object_name: str, file_path: Path) -> None:
        bucket = self.minioClient.bucket_exists(bucket_name)
        if bucket:
            self.minioClient.remove_object(bucket_name, object_name)
            self.minioClient.fput_object(bucket_name, object_name, file_path)
            print(f"File {object_name} has been successfully updated")
