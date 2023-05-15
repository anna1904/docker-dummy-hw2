from pathlib import Path

from datasets import load_dataset
from datasets.arrow_reader import ArrowReader
from sklearn.model_selection import train_test_split


def load_mnist_data(path_to_save: Path, random_state: int = 42):

    path_to_save.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("mnist", split=f"train[:{5000}]")

    df_train, df_rest = train_test_split(dataset, test_size= 0.2, random_state=random_state)
    df_test, df_val = train_test_split(df_rest, test_size = 0.5, random_state=random_state)

    df_train.to_csv(path_to_save / "train.csv", index=False)
    df_val.to_csv(path_to_save / "val.csv", index=False)
    df_test.to_csv(path_to_save / "test.csv", index=False)