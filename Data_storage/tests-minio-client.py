import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from minio import Minio

from client_minio import Minio_client, ACCESS_KEY, SECRET_KEY, ENDPOINT

class TestMinioClient(unittest.TestCase):

    def setUp(self):
        # Set up a mock Minio client object for testing
        self.mock_client = MagicMock(spec=Minio)
        self.client = Minio_client()
        self.client.minioClient = self.mock_client

    def test_upload_object(self):
        # Set up test data
        bucket_name = "test-bucket"
        object_name = "test-object"
        file_path = Path("test-file.txt")

        # Test successful upload
        self.mock_client.bucket_exists.return_value = True
        self.client.upload_object(bucket_name, object_name, file_path)
        self.mock_client.fput_object.assert_called_once_with(
            bucket_name, object_name, file_path)

    def test_get_object(self):
        # Set up test data
        bucket_name = "test-bucket"
        object_name = "test-object"
        file_path = Path("test-file.txt")

        # Test successful get
        self.mock_client.bucket_exists.return_value = True
        self.mock_client.fget_object.return_value = MagicMock(
            spec=object, object_name=object_name)
        result = self.client.get_object(bucket_name, object_name, file_path)
        self.mock_client.fget_object.assert_called_once_with(
            bucket_name, object_name, file_path)
        self.assertEqual(result.object_name, object_name)


    def test_remove_object(self):
        # Set up test data
        bucket_name = "test-bucket"
        object_name = "test-object"

        # Test successful removal
        self.mock_client.bucket_exists.return_value = True
        self.client.remove_object(bucket_name, object_name)
        self.mock_client.remove_object.assert_called_once_with(
            bucket_name, object_name)


    def test_update_object(self):
        # Set up test data
        bucket_name = "test-bucket"
        object_name = "test-object"
        file_path = Path("test-file.txt")

        # Test successful update
        self.mock_client.bucket_exists.return_value = True
        self.client.update_object(bucket_name, object_name, file_path)
        self.mock_client.remove_object.assert_called_once_with(
            bucket_name, object_name)
        self.mock_client.fput_object.assert_called_once_with(
            bucket_name, object_name, file_path)

if __name__ == '__main__':
    unittest.main()
