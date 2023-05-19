from datasets import load_dataset
import evaluate
import numpy as np
import wandb
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    DefaultDataCollator
)
import sys
import os
from classification.config import DataTrainingArguments

# def get_args():
#     parser = HfArgumentParser((DataTrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         data_args = parser.parse_args_into_dataclasses()
#     return data_args

def get_config(config_path: Path):
    parser = HfArgumentParser(DataTrainingArguments)
    [data_args] = parser.parse_json_file(config_path)
    return data_args


def read_dataset(data_args: DataTrainingArguments):
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    mnist = load_dataset("csv", data_files=data_files)

    labels = mnist["train"].unique("label")
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return mnist, labels, label2id, id2label

def get_model():
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return checkpoint, image_processor

def process_dataset(mnist):
    checkpoint, image_processor = get_model()
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    mnist = mnist.with_transform(transforms)

    return mnist

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    # wandb.log({"my_accuracy": acc})
    return acc


def training(config_path: Path):
    data_args = get_config(config_path)
    mnist, labels, label2id, id2label = read_dataset(data_args)
    checkpoint, image_processor = get_model()
    dataset = process_dataset(mnist)
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # wandb.init(
    #     project="classification_example",
    #     name="sweep-3",
    #     config={
    #         "epochs": 0.1,
    #     })

    training_args = TrainingArguments(
        output_dir="my_classification_model",
        use_fast_tokenizer= True,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        num_train_epochs=wandb.config.epochs,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        # report_to="wandb",
        max_steps=2,
        logging_steps=1,  # we will log every 100 steps
        eval_steps=1,  # we will perform evaluation every 1 steps
        eval_accumulation_steps=1,  # report evaluation results after each step
        load_best_model_at_end=True,
        save_steps = 1,
        do_eval=True,
        weight_decay= 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    # wandb.log({'constant': 0.9})
    trainer.train()

