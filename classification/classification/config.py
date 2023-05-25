from dataclasses import dataclass


@dataclass
class DataTrainingArguments:
    train_file: str
    evaluation_file: str
