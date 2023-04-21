from minio import Minio
import unittest

class Minio_client():
    def __init__(self):
        try:
            self.minioClient = Minio("hostPath", access_key = "db_user", secret_key = "db_key", secure = False)
        except:
            print("Not able to connect minio / {}".format(Exception))

    def create_object(self, bucket_name, object_name):
        try:
            bucket = self.minioClient.bucket_exists(bucket_name)
            if bucket:
                self.minioClient.put_object(bucket_name, object_name, data=object_name)
                print("Dataset/Model has been sucessfully saved in {}".format(bucket_name))


class UnitTests(unittest.TestCase)
    __init__(self):




