import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from minio import Minio

from client_minio import MinioClient


@pytest.fixture()
def bucket_name() -> str:
    return "test-bucket"


@pytest.fixture()
def object_name() -> str:
    return "test-object"


@pytest.fixture()
def file_path() -> Path:
    return Path("test-file.txt")


class TestMinioClient(unittest.TestCase):
    def setUp(self) -> None:
        # Set up a mock Minio client object for testing
        self.mock_client = MagicMock(spec=Minio)
        self.client = MinioClient()
        self.client.minioClient = self.mock_client

    def test_upload_object(self) -> None:
        # Test successful upload
        self.mock_client.bucket_exists.return_value = True
        self.client.upload_object(bucket_name, object_name, file_path)
        self.mock_client.fput_object.assert_called_once_with(
            bucket_name, object_name, file_path)

    def test_get_object(self) -> None:
        # Test successful get
        self.mock_client.bucket_exists.return_value = True
        self.mock_client.fget_object.return_value = MagicMock(
            spec=object, object_name=object_name)
        result = self.client.get_object(bucket_name, object_name, file_path)
        self.mock_client.fget_object.assert_called_once_with(
            bucket_name, object_name, file_path)
        self.assertEqual(result.object_name, object_name)

    def test_remove_object(self) -> None:
        # Test successful removal
        self.mock_client.bucket_exists.return_value = True
        self.client.remove_object(bucket_name, object_name)
        self.mock_client.remove_object.assert_called_once_with(
            bucket_name, object_name)

    def test_update_object(self) -> None:
        # Test successful update
        self.mock_client.bucket_exists.return_value = True
        self.client.update_object(bucket_name, object_name, file_path)
        self.mock_client.remove_object.assert_called_once_with(
            bucket_name, object_name)
        self.mock_client.fput_object.assert_called_once_with(
            bucket_name, object_name, file_path)


if __name__ == '__main__':
    unittest.main()
