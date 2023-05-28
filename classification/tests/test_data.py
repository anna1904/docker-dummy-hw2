import pytest
import torch
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@pytest.fixture
def mnist_dataset():
    return load_dataset("mnist")

def test_mnist_dataset_shape(mnist_dataset):
    example = mnist_dataset["train"][0]
    assert example["label"] >= 0 and example["label"] <= 9

def test_mnist_dataset_type(mnist_dataset):
    example = mnist_dataset["train"][0]
    image = np.array(example["image"])
    assert isinstance(image, np.ndarray)
    assert isinstance(example["label"], int)


def test_mnist_dataset_download():
    with pytest.raises(ValueError):
        load_dataset("mnist", split="nonexistent_split")


