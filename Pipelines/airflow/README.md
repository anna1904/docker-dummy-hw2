1. Run standalone airflow

``export AIRFLOW_HOME=$PWD/airflow-home
airflow standalone``

2. Configure airlfow <> k8s connection
3. Create storage
   ``kubectl create -f airflow-volumes.yaml``
4. Run pipelines
   https://madewithml.com/courses/mlops/orchestration/