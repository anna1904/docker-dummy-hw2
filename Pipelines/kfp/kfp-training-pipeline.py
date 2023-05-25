import os
import uuid
from typing import Optional
from kfp.onprem import use_k8s_secret

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
        file_outputs={"train": "/tmp/data/train.ds", "test": "/tmp/data/test.ds"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    train_model = dsl.ContainerOp(
        name="train_model ",
        command="python classification/cli.py training ./conf/config_test.json".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(load_data.outputs["train"], path="/tmp/data/train.ds"),
            dsl.InputArgumentPath(load_data.outputs["test"], path="/tmp/data/test.ds"),
        ],
        file_outputs={
            "config": "/tmp/results/config.json",
            "model": "/tmp/results/pytorch_model.bin",
            "all_results": "/tmp/results/all_results.json",
            "preprocessor_config": "/tmp/results/preprocessor_config.json",
            "eval_results": "/tmp/results/eval_results.json",
            "train_results": "/tmp/results/train_results.json",
            "trainer_state": "/tmp/results/trainer_state.json",
            "training_args": "/tmp/results/training_args.bin"
        },
    )
    train_model.set_memory_request('2G').set_memory_limit('4G').set_cpu_request('4').set_cpu_limit('8')

    upload_model = dsl.ContainerOp(
        name="upload_model ",
        command="python classification/cli.py upload-to-registry classification_example /tmp/results WANDB_PROJECT".split(),
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(train_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(train_model.outputs["model"], path="/tmp/results/pytorch_model.bin"),
            dsl.InputArgumentPath(train_model.outputs["all_results"], path="/tmp/results/all_results.json"),
            dsl.InputArgumentPath(train_model.outputs["preprocessor_config"],
                                  path="/tmp/results/preprocessor_config.json"),
            dsl.InputArgumentPath(
                train_model.outputs["eval_results"], path="/tmp/results/eval_results.json"
            ),
            dsl.InputArgumentPath(train_model.outputs["train_results"], path="/tmp/results/train_results.json"),
            dsl.InputArgumentPath(train_model.outputs["trainer_state"], path="/tmp/results/trainer_state.json"),
            dsl.InputArgumentPath(train_model.outputs["training_args"], path="/tmp/results/training_args.bin"),
        ],
    )

    train_model.apply(use_k8s_secret(secret_name='wb-credentials',
                                     k8s_secret_key_to_env={'WANDB_PROJECT': 'WANDB_PROJECT',
                                                            'WANDB_API_KEY': 'WANDB_API_KEY'}))

    upload_model.apply(use_k8s_secret(secret_name='wb-credentials',
                                      k8s_secret_key_to_env={'WANDB_PROJECT': 'WANDB_PROJECT',
                                                             'WANDB_API_KEY': 'WANDB_API_KEY'}))


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
