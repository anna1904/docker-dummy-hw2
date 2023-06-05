from dataclasses import dataclass


@dataclass
class DataTrainingArguments:
    train_dir: str
    evaluation_dir: str
