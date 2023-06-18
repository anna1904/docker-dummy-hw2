import datasets
from datasets import load_metric
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
from classification.config import DataTrainingArguments
from alibi_detect.cd import MMDDrift
from alibi_detect.saving import save_detector, load_detector


def get_config(config_path: Path):
    parser = HfArgumentParser(DataTrainingArguments)
    [data_args] = parser.parse_json_file(config_path)
    return data_args


def read_dataset(data_args: DataTrainingArguments):
    mnist = datasets.load_from_disk(data_args.train_file)

    labels = mnist.unique("label")
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return mnist, labels, label2id, id2label


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
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)

    return dataset


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    return {'accuracy': metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1),
                                       references=eval_pred.label_ids)}


def training(config_path: Path):
    data_args = get_config(config_path)
    mnist, labels, label2id, id2label = read_dataset(data_args)
    checkpoint, image_processor = get_model()
    dataset = process_dataset(mnist)
    dataset = dataset.train_test_split(0.001)
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    wandb.init(
        project="classification_example",
        name="test_23-05",
        dir="/tmp",
        config={
            "epochs": 1,
        })

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
        push_to_hub=False,
        report_to="wandb",
        # logging_steps=100,
        do_eval=True,
        weight_decay=0)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Data drift detector
    detector = MMDDrift(dataset["train"], backend='pytorch', p_val=.05)
    save_detector(detector, data_args.detector_path)

    # Evaluation
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    metrics["eval_samples"] = len(dataset["test"])
    trainer.save_metrics("eval", metrics)
