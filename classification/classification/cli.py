import typer

from classification.trainer import training
from classification.utils import upload_to_registry

app = typer.Typer()
app.command()(training)
app.command()(upload_to_registry)
# app.command()(load_from_registry)

if __name__ == "__main__":
    app()