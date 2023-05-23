import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar

IMAGE = "anko47/classification-app:latest"
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


@dsl.pipeline(name="classification_traininig_pipeline", description="classification_traininig_pipeline")
def classification_traininig_pipeline():

    load_data = dsl.ContainerOp(
        name="load_data",
        command="python classification/cli.py load-mnist-data /tmp/data/".split(),
        image=IMAGE,
        file_outputs={"train": "/data/train.csv", "val": "/data/val.csv", "test": "/data/test.csv"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    train_model = dsl.ContainerOp(
        name="train_model ",
        command="python classification/cli.py training".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(load_data.outputs["train"], path="/data/train.csv"),
            dsl.InputArgumentPath(load_data.outputs["val"], path="/data/val.csv"),
            dsl.InputArgumentPath(load_data.outputs["test"], path="/data/test.csv"),
        ],
        file_outputs={
            "config": "/tmp/results/config.json",
            "model": "/tmp/results/pytorch_model.bin",
            "optimizer": "/tmp/results/optimizer.pt",
            "preprocessor_config": "/tmp/results/preprocessor_config.json",
            "rng_state": "/tmp/results/rng_state.pth",
            "trainer_state": "/tmp/results/trainer_state.json",
            "training_args": "/tmp/results/training_args.bin"
        },
    )

    upload_model = dsl.ContainerOp(
        name="upload_model ",
        command="python classification/cli.py upload-to-registry kfp-pipeline /tmp/results WANDB_PROJECT".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(train_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(train_model.outputs["model"], path="/tmp/results/pytorch_model.bin"),
            dsl.InputArgumentPath(train_model.outputs["optimizer"], path="/tmp/results/optimizer.json"),
            dsl.InputArgumentPath(train_model.outputs["preprocessor_config"], path="/tmp/results/preprocessor_config.json"),
            dsl.InputArgumentPath(
                train_model.outputs["rng_state"], path="/tmp/results/rng_state.pth"
            ),
            dsl.InputArgumentPath(train_model.outputs["trainer_state"], path="/tmp/results/trainer_state.json"),
            dsl.InputArgumentPath(train_model.outputs["training_args"], path="/tmp/results/training_args.bin"),
        ],
    )

    env_var_project = V1EnvVar(name="WANDB_PROJECT", value=WANDB_PROJECT)
    upload_model = upload_model.add_env_variable(env_var_project)

    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=WANDB_API_KEY)
    upload_model = upload_model.add_env_variable(env_var_password)


def compile_pipeline() -> str:
    path = "/tmp/classification_traininig_pipeline.yaml"
    kfp.compiler.Compiler().compile(classification_traininig_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline")
    name = "classification-sample-training"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version.")
        pipeline_prev_version = client.get_pipeline(client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.id,
        )
    else:
        pipeline = client.upload_pipeline(pipeline_package_path=compile_pipeline(), pipeline_name=name)
    print(f"pipeline {pipeline.id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)