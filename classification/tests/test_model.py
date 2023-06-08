from pathlib import Path

from typing import Dict
import pytest
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

from classification.trainer import training, compute_metrics, process_dataset, read_dataset, get_model
from classification.config import DataTrainingArguments


@pytest.fixture()
def data_path() -> Path:
    return Path("classification/data/")


@pytest.fixture()
def data_args(data_path: Path) -> DataTrainingArguments:
    return DataTrainingArguments(
        train_file=str(data_path / "train.ds"),
        evaluation_file=str(data_path / "test.ds")
    )


# Overfit on a batch
@pytest.fixture()
def trainer_with_one_batch(data_args: DataTrainingArguments) -> Trainer:
    mnist, labels, label2id, id2label = read_dataset(data_args)
    checkpoint, image_processor = get_model()
    dataset = process_dataset(mnist)
    dataset = dataset.train_test_split(0.1)
    train_set = dataset['train'].select(range(1))
    eval_set = dataset['test'].select(range(1))
    dataset = {"train": train_set, "test": eval_set}
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="my_classification_model",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-03,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        num_train_epochs=100000,
        max_steps=400,
        push_to_hub=False,
        logging_steps=100,  # we will log every 100 steps
        eval_steps=50000,
        eval_accumulation_steps=1,  # report evaluation results after each step
        load_best_model_at_end=False,
        save_steps=500,
        do_eval=False,
        seed=42,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=image_processor
    )
    return trainer


def test_overfit_batch(trainer_with_one_batch: Trainer):
    train_result = trainer_with_one_batch.train()
    metrics = train_result.metrics
    assert metrics["train_loss"] < 0.01
