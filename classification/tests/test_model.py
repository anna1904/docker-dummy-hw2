from pathlib import Path

import pytest
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

from classification.trainer import training, compute_metrics, process_dataset, read_dataset, get_model
# @pytest.fixture()
# def data_args(data_path: Path) -> DataTrainingArguments:
#     return DataTrainingArguments(
#         train_file=str(data_path / "train.csv"),
#         validation_file=str(data_path / "val.csv"),
#         max_train_samples=4,
#         max_eval_samples=2,
#     )
#Overfit on a batch
@pytest.fixture()
def trainer_with_one_batch() -> Trainer:

    mnist, labels, label2id, id2label = read_dataset(size=10)
    checkpoint, image_processor = get_model()
    dataset = process_dataset(mnist)
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="my_classification_model",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-04,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1000,
        push_to_hub=False,
        max_steps=1,
        logging_steps=1,  # we will log every 100 steps
        eval_steps=1,  # we will perform evaluation every 1 steps
        eval_accumulation_steps=1,  # report evaluation results after each step
        load_best_model_at_end=True,
        save_steps=1,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor
    )
    return trainer

def test_overfit_batch(trainer_with_one_batch: Trainer):
    train_result = trainer_with_one_batch.train()
    metrics = train_result.metrics
    assert metrics["train_loss"] < 0.01