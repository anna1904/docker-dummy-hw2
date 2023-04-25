from minio import Minio
from pathlib import Path
import unittest

ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
ENDPOINT = "0.0.0.0:9000"

class Minio_client():
    def __init__(self):
        try:
            self.minioClient = Minio(ENDPOINT, access_key = ACCESS_KEY, secret_key = SECRET_KEY, secure = False)
        except:
            print("Not able to connect minio / {}".format(Exception))

    def upload_object(self, bucket_name, object_name, file_path: Path):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.fput_object(bucket_name, object_name, file_path)
                print("File has been sucessfully saved in {}".format(bucket_name))
        except:
            print("Not able to upload file / {}".format(Exception))
    def get_object(self, bucket_name, object_name, file_path: Path):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                result = self.minioClient.fget_object(bucket_name, object_name, file_path)
                print("res:", result)
                print(f"File {result.object_name} has been sucessfully get from {bucket_name}")
                return result
        except Exception as e:
            print("Cannot upload an object/ {}".format(e))
    def remove_object(self, bucket_name, object_name):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.remove_object(bucket_name, object_name)
                print("File has been sucessfully deleted from {}".format(bucket_name))
        except:
            print("Cannot delete an object/ {}".format(Exception))

    #As objects are immutable in S3, I first delete the object and upload the new version with the same name
    def update_object(self, bucket_name, object_name, file_path):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.remove_object(bucket_name, object_name)
                self.minioClient.fput_object(bucket_name, object_name, file_path)
                print(f"File {object_name} has been sucessfully updated")
        except:
            print("Cannot update an object/ {}".format(Exception))
c = Minio_client()
c.upload_object('mybucket','qw', 'qw.txt' )
c.get_object('mybucket','qw', 'qw.txt' )









