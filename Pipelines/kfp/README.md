1. Create kind cluster 

``kind create cluster --name kubeflow-pipelines-standalone``

2. Run your cluster
``k9s -A``

3. Deploy Kuberflow pipelines

``
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
``

4. Get the public URL for the Kubeflow Pipelines UI and use it to access the Kubeflow Pipelines UI

``kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com
``

Or by yaml
`` 
export PIPELINE_VERSION=1.8.5
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" > kfp-yml/res.yaml
kubectl create -f kfp-yml/res.yaml

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION" > kfp-yml/pipelines.yaml
kubectl create -f kfp-yml/pipelines.yaml

``

5. Access UI and minio

``
kubectl port-forward --address=0.0.0.0 svc/minio-service 9000:9000 -n kubeflow
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow
``

6. To login MINIO take thigs from yaml
7. 

