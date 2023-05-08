from datasets import load_dataset
import evaluate
import numpy as np
import wandb
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)


def read_dataset():
    food = load_dataset("mnist", split="train[:5000]")
    food = food.train_test_split(test_size=0.2)

    labels = food["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return food, labels, label2id, id2label

def get_model():
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return checkpoint, image_processor

def process_dataset(food):
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

    food = food.with_transform(transforms)

    return food

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    wandb.log({"my_accuracy": acc})
    return acc


def training():
    food, labels, label2id, id2label = read_dataset()
    checkpoint, image_processor = get_model()
    dataset = process_dataset(food)
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    wandb.init(
        project="classification_example",
        name="sweep-3",
        config={
            "epochs": 0.1,
        })
    config = wandb.config

    training_args = TrainingArguments(
        output_dir="my_classification_model",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        num_train_epochs=config.epochs,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="wandb",
        max_steps=2,
        logging_steps=1,  # we will log every 100 steps
        eval_steps=1,  # we will perform evaluation every 1 steps
        eval_accumulation_steps=1,  # report evaluation results after each step
        load_best_model_at_end=True,
        save_steps = 1,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )
    wandb.log({'constant': 0.9})
    trainer.train()
