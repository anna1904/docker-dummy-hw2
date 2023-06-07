0. Install airflow-apache, apache-airflow-providers-cncf-kubernetes
1. kind create cluster --name cluster-a
2. ks9 -A
3. Run standalone airflow

``export AIRFLOW_HOME=$PWD/airflow-home
airflow standalone``

4. Save the credentials from output

5. Configure airlfow <> k8s connection
   Admin -> Connections -> Kubernetes: Edit Write Kubeconfig Path : PWD + /.kube/config

6. Add dags
   Got to airflow home directory, create folders 'dags' and put there your python files

7. Create storage (volume to pass data from one pod to another)
   ``kubectl create -f airflow-volumes.yaml``
8. Run dags

Note: In Kuberflow pipelines you created Minio where you put your data and transfer,
here you create volumes in kubernetes

9. Add secrets
   ``airflow connections add my_secret --conn-uri 'your_secret_value'``