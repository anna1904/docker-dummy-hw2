from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import \
    KubernetesPodOperator
from kubernetes.client import models as k8s

volume = k8s.V1Volume(
    name="inference-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="inference-storage"),
)
volume_mount = k8s.V1VolumeMount(name="inference-storage", mount_path="/tmp/model/", sub_path=None)

with DAG(start_date=datetime(2021, 1, 1), catchup=False, schedule_interval=None, dag_id="inference_dag") as dag:
    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image="anko47/classification-app:latest",
        cmds=["rm", "-rf", "/tmp/model/data/*"],
        task_id="clean_storage_before_start",
        in_cluster=False,
        namespace="default",
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

    load_model = KubernetesPodOperator(
        name="load_model",
        image="anko47/classification-app:latest",
        cmds=["python", "classification/cli.py", "load-from-registry", "classification_example:latest",
              "/tmp/model/results/", "classification_example"],
        task_id="load_model",
        env_vars={"WANDB_PROJECT": "classification_example",
                  "WANDB_API_KEY": "***"},
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )
    detect_data_drift = KubernetesPodOperator(
        name="detect_data_drift",
        image="anko47/classification-app:latest",
        cmds=[
            "python",
            "classification/cli.py",
            "detect-data-drift",
            "/tmp/model/data/test.ds",
            "/tmp/model/data/detector/detector.pkl",
        ],
        task_id="detect_data_drift",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    run_inference = KubernetesPodOperator(
        name="run_inference",
        image="anko47/classification-app:latest",
        cmds=[
            "python",
            "classification/cli.py",
            "run-inference-on-ds",
            "/tmp/model/data/test.ds",
            "/tmp/model/results/",
            "/tmp/model/inference",
        ],
        task_id="run_inference",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image="anko47/classification-app:latest",
        cmds=["rm", "-rf", "/tmp/model/data/*"],
        task_id="clean_up",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data
    clean_storage_before_start >> load_model

    load_data >> run_inference
    load_model >> run_inference
    run_inference >> clean_up
