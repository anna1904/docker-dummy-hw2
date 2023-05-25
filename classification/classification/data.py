from pathlib import Path

from datasets import load_dataset


def load_mnist_data(path_to_save: Path):
    path_to_save.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("mnist", split="train[:500]")
    test_dataset = load_dataset("mnist", split="test")

    dataset.save_to_disk(path_to_save / "train.ds")
    test_dataset.save_to_disk(path_to_save / "test.ds")
