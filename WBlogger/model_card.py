from huggingface_hub import ModelCard, ModelCardData
import wandb
from pathlib import Path

# Training code goes here...

# Initialize ModelCardData
card_data = ModelCardData(
    name="MNIST Image Classification",
    language="en",
    license="MIT",
    model_type="image-classification",
    dataset_tags=["mnist"],
    model_tags=["vit"],
)

# Create ModelCard
card = ModelCard.from_template(
    card_data,
    model_id="mnist-image-classification-vit",
    model_description="Image classification model trained on the MNIST dataset using the ViT (Vision Transformer) architecture.",
    training_data="./train.py",
    citation="cite ViT: Vision Transfomer article ",
    authors=["Anna Konovalenko"],
    codebase="https://github.com/your_username/mnist-image-classification",
    version="v1.0",
)

# Add artifacts to ModelCard
with wandb.init() as _:
    run = wandb.run
    model_artifact = wandb.Artifact("model", type="model", description="MNIST image classification model")
    model_artifact.add_dir("my_classification_model")
    run.log_artifact(model_artifact)
    card.artifacts = [{"artifactName": model_artifact.name, "type": model_artifact.type}]

# Save ModelCard
card.save("model_card.md")
print(card)
