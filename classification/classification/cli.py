import typer

from classification.data import load_data
from classification.trainer import training
from classification.utils import upload_to_registry, load_from_registry
from classification.predictor import run_inference_on_ds

app = typer.Typer()
app.command()(training)
app.command()(load_data)
app.command()(upload_to_registry)
app.command()(run_inference_on_ds)
app.command()(load_from_registry)

if __name__ == "__main__":
    app()
