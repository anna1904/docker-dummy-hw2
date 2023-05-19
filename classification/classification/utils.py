import os

import wandb
from pathlib import Path

def upload_to_registry(model_name: str, model_path: Path, project_name: str):
    with wandb.init(project=project_name) as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "config.json")
        art.add_file(model_path / "pytorch_model.bin")
        art.add_file(model_path / "optimizer.pt")
        art.add_file(model_path / "preprocessor_config.json")
        art.add_file(model_path / "rng_state.pth")
        art.add_file(model_path / "scheduler.pt")
        art.add_file(model_path / "trainer_state.json")
        art.add_file(model_path / "training_args.bin")
        wandb.log_artifact(art)

def load_from_registry(model_name: str, model_path: Path, project_name: str):
    with wandb.init(project_name) as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")
