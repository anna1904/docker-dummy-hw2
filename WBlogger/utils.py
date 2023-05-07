import wandb
from pathlib import Path

def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "config.json")
        art.add_file(model_path / "pytorch_model.bin")
        art.add_file(model_path / "tokenizer.json")
        art.add_file(model_path / "tokenizer_config.json")
        art.add_file(model_path / "special_tokens_map.json")
        art.add_file(model_path / "README.md")
        wandb.log_artifact(art)