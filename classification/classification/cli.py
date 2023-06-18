import typer

from classification.data import load_mnist_data
from classification.trainer import training
from classification.utils import upload_to_registry, load_from_registry
from classification.predictor import run_inference_on_ds, detect_data_drift

app = typer.Typer()
app.command()(training)
app.command()(load_mnist_data)
app.command()(upload_to_registry)
app.command()(run_inference_on_ds)
app.command()(detect_data_drift)
app.command()(load_from_registry)

if __name__ == "__main__":
    app()
