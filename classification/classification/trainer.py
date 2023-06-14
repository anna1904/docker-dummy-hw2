import datasets
import torch
from datasets import load_metric
import os
from torchvision.datasets.folder import ImageFolder
from datasets import load_dataset
import numpy as np
import wandb
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from torch.utils.data import random_split
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    DefaultDataCollator
)
from config import DataTrainingArguments

MANUAL_SEED = 42


def get_config(config_path: Path):
    parser = HfArgumentParser(DataTrainingArguments)
    [data_args] = parser.parse_json_file(config_path)
    return data_args


def read_dataset(data_args: DataTrainingArguments):
    dataset = load_dataset("imagefolder", data_dir=data_args.train_dir)
    # dataset = dataset.shuffle(seed=MANUAL_SEED)
    labels = dataset.unique("label")
    label2id, id2label = dict(), dict()
    for i in labels["train"]:
        label2id[f"{i}_label"] = i
        id2label[i] = f"{i}_label"

    return dataset, labels, label2id, id2label


def get_model():
    checkpoint = "facebook/deit-tiny-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return checkpoint, image_processor


def process_dataset(dataset):
    checkpoint, image_processor = get_model()
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")).float() for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)

    return dataset


def compute_metrics(eval_pred):
    metric_accuracy = load_metric("accuracy")
    metric_precision = load_metric("precision")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    predictions = predictions.astype(np.float32)  # Convert predictions to Float type

    accuracy = metric_accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    precision = metric_precision.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')

    return {'accuracy': accuracy, 'precision': precision}


def training(config_path: Path):
    torch.manual_seed(MANUAL_SEED)
    data_args = get_config(config_path)
    dataset, labels, label2id, id2label = read_dataset(data_args)
    checkpoint, image_processor = get_model()
    dataset = process_dataset(dataset['train'])

    # Split dataset into train and evaluation sets
    train_size = int(len(dataset) * 0.9)  # Adjust the split ratio as needed
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels["train"]),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    wandb.init(
        project="classification-losses",
        dir="/tmp",
        config={
            "epochs": 10,
        })
    # Specify the output directory for saving the model
    # output_dir = "./model-losses"  # Use a directory in your local file system
    # os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir="/tmp/model/model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-05,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=10,
        num_train_epochs=wandb.config.epochs,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="wandb",
        # logging_steps=100,
        do_eval=True,
        weight_decay=0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.save_metrics("eval", metrics)


training(Path('/Users/anko/Development/Projector/docker-dummy-hw2/classification/conf/config.json'))
