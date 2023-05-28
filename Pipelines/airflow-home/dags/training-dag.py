from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

volume = k8s.V1Volume(name="training-storage",
                      persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="training-storage"), )
volume_mount = k8s.V1VolumeMount(name="training-storage", mount_path="/tmp/model/", sub_path=None)

with DAG(start_date=datetime(2021, 1, 1), catchup=False, schedule_interval=None, dag_id="training_dag") as dag:
    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image="anko47/classification-app:latest",
        cmds=["rm", "-rf", "/tmp/model/*"],
        task_id="clean_storage_before_start",
        in_cluster=False,
        namespace="default",
        startup_timeout_seconds=900,
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    load_data = KubernetesPodOperator(
        name="load_data",
        image="anko47/classification-app:latest",
        cmds=["python", "classification/cli.py", "load-mnist-data", "/tmp/model/data/"],
        task_id="load_data",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    train_model = KubernetesPodOperator(
        name="train_model",
        image="anko47/classification-app:latest",
        cmds=["python", "classification/cli.py", "training", "./conf/config_test.json"],
        task_id="train_model",
        env_vars={"WANDB_PROJECT": "classification_example",
                  "WANDB_API_KEY": "***"},
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_model = KubernetesPodOperator(
        name="upload_model",
        image="anko47/classification-app:latest",
        cmds=["python", "classification/cli.py", "upload-to-registry", "classification_example", "/tmp/model/model",
              "classification_example"],
        task_id="upload_model",
        env_vars={"WANDB_PROJECT": "classification_example",
                  "WANDB_API_KEY": "***"},
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image="anko47/classification-app:latest",
        cmds=["rm", "-rf", "/tmp/model/*"],
        task_id="clean_up",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data >> train_model >> upload_model >> clean_up
