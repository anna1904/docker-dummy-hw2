import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kfp.onprem import use_k8s_secret

IMAGE = "anko47/classification-app:latest"


@dsl.pipeline(name="classification_inference_pipeline", description="classification_inference_pipeline")
def classification_inference_pipeline():
    load_data = dsl.ContainerOp(
        name="load_data",
        command="python classification/cli.py load-mnist-data /tmp/data/".split(),
        image=IMAGE,
        file_outputs={"test": "/tmp/data/test.ds"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    load_model = dsl.ContainerOp(
        name="load_model",
        command="python classification/cli.py load-from-registry classification_example:latest /tmp/results "
                "classification_example".split(),
        image=IMAGE,
        file_outputs={
            "config": "/tmp/results/config.json",
            "model": "/tmp/results/pytorch_model.bin",
            "preprocessor_config": "/tmp/results/preprocessor_config.json",
            "trainer_state": "/tmp/results/trainer_state.json",
            "training_args": "/tmp/results/training_args.bin"
        }
    )
    load_model.execution_options.caching_strategy.max_cache_staleness = "P0D"

    run_inference = dsl.ContainerOp(
        name="run_inference",
        command="python classification/cli.py run-inference-on-ds /tmp/data/test.ds /tmp/results/ /tmp/inference".split(),
        image=IMAGE,
        artifact_argument_paths=
        [
            dsl.InputArgumentPath(load_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(load_model.outputs["model"], path="/tmp/results/pytorch_model.bin"),
            dsl.InputArgumentPath(load_model.outputs["preprocessor_config"],
                                  path="/tmp/results/preprocessor_config.json"),
            dsl.InputArgumentPath(load_model.outputs["trainer_state"], path="/tmp/results/trainer_state.json"),
            dsl.InputArgumentPath(load_model.outputs["training_args"], path="/tmp/results/training_args.bin"),
            dsl.InputArgumentPath(load_data.outputs["test"], path="/tmp/data/test.ds"),
        ],
        file_outputs={
            "pred": "/tmp/inference/",
        },
    )

    load_model.apply(use_k8s_secret(secret_name='wb-credentials',
                                    k8s_secret_key_to_env={'WANDB_PROJECT': 'WANDB_PROJECT',
                                                           'WANDB_API_KEY': 'WANDB_API_KEY'}))


def compile_pipeline() -> str:
    path = "/tmp/nlp_inference_pipeline.yaml"
    kfp.compiler.Compiler().compile(classification_inference_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("inference", namespace=namespace)

    print("Uploading pipeline")
    name = "classification_inference_pipeline"
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
