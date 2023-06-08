import pytest
import torch
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classification.config import DataTrainingArguments
from pathlib import Path
import datasets


@pytest.fixture
def mnist_dataset():
    return load_dataset("mnist")


@pytest.fixture()
def data_path() -> Path:
    return Path("classification/data/")


@pytest.fixture()
def data_args(data_path: Path) -> DataTrainingArguments:
    return DataTrainingArguments(
        train_file=str(data_path / "train.ds"),
        evaluation_file=str(data_path / "test.ds")
    )


def test_mnist_dataset_shape(data_args: DataTrainingArguments):
    mnist = datasets.load_from_disk(data_args.train_file)
    labels = mnist["label"]
    assert all(0 <= label <= 9 for label in labels)


def test_mnist_dataset_type(mnist_dataset):
    example = mnist_dataset["train"][0]
    image = np.array(example["image"])
    assert isinstance(image, np.ndarray)
    assert isinstance(example["label"], int)


def test_mnist_dataset_download():
    with pytest.raises(ValueError):
        load_dataset("mnist", split="nonexistent_split")
