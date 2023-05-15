import typer

from classification.utils import load_mnist_data
from classification.trainer import training
from classification.utils import upload_to_registry

app = typer.Typer()
app.command()(training)
app.command()(load_mnist_data)
app.command()(upload_to_registry)
# app.command()(load_from_registry)

if __name__ == "__main__":
    app()