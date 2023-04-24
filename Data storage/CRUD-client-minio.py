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

    def upload_object(self, bucket_name, file_path: Path):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.fput_object(bucket_name, file_path.name, file_path)
                print("Dataset/Model has been sucessfully saved in {}".format(bucket_name))
        except:
            print('Cannot upload an object')
    def get_object(self, bucket_name, file_path: Path):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.fget_object(bucket_name, file_path.name, file_path)
                print("Dataset/Model has been sucessfully saved in {}".format(bucket_name))
        except:
            print('Cannot upload an object')

    # def delete_object(self, bucket_name, file_path: Path):
    #     try:
    #         bucket = self.minioClient.bucket_exists(bucket_name)
    #         if bucket:
    #             self.minioClient.fput_object(bucket_name, file_path.name, file_path)
    #             print("Dataset/Model has been sucessfully saved in {}".format(bucket_name))
    #     except:
    #         print('Cannot upload an object')
    # def update_object(self, bucket_name, file_path: Path):
    #     try:
    #         bucket = self.minioClient.bucket_exists(bucket_name)
    #         if bucket:
    #             self.minioClient.fput_object(bucket_name, file_path.name, file_path)
    #             print("Dataset/Model has been sucessfully saved in {}".format(bucket_name))
    #     except:
    #         print('Cannot upload an object')






