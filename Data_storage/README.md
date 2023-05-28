# Local Deployment

https://min.io/docs/minio/macos/index.html

For arm42:
#### MNIO:

1. Install minio
```shell
    brew install minio/stable/minio
```

```shell
    minio server start
    
    MinIO Object Storage Server
Copyright: 2015-2023 MinIO, Inc.
License: GNU AGPLv3 <https://www.gnu.org/licenses/agpl-3.0.html>
Version: RELEASE.2023-04-20T17-56-55Z (go1.20.3 darwin/arm64)

Status:         1 Online, 0 Offline.
S3-API: http://192.168.1.25:9000  http://127.0.0.1:9000
RootUser: minioadmin
RootPass: minioadmin

Console: http://192.168.1.25:55481 http://127.0.0.1:55481
RootUser: minioadmin
RootPass: minioadmin
```
2. Run the local server from the address: http://127.0.0.1:9000 and log in with credintials
(MinIO automatically redirects browser access to the MinIO Console)


3. Now you can create buckets, upload your files there and create service accounts

### Install client

```shell
    brew install minio/stable/mc
```
The MinIO Client allows you to work with your MinIO volume from the commandline.

# DOCKER
https://min.io/docs/minio/container/index.html

0. Run Docker Daemon
1. Start the container
```angular2html
mkdir -p ~/minio/data

docker run \
   -p 9000:9000 \
   -p 9090:9090 \
   --name minio \
   -v ~/minio/data:/data \
   -e "MINIO_ROOT_USER=ROOTNAME" \
   -e "MINIO_ROOT_PASSWORD=CHANGEME123" \
   quay.io/minio/minio server /data --console-address ":9090"
```
 2. Log in and the same as with local development.
# KUBERNETES

https://min.io/docs/minio/kubernetes/upstream/index.html
and standalone: https://github.com/kubernetes/examples/tree/master/staging/storage/minio#minio-standalone-server-deployment

1. Download yaml with kubernetes setting
```angular2html
curl https://raw.githubusercontent.com/minio/docs/master/source/extra/examples/minio-dev.yaml -O

or create own by standalone yaml with PVC
```
2. Create cluster 
```
kind create cluster --name hw3_cluster
```
3. Check containers ```k9s -A```
4. Apply yaml
```angular2html
kubectl apply -f minio-standalone.yaml
--------------
pod/minio created
persistentvolumeclaim/minio-pv-claim created
```
5.Verify the state
```angular2html
kubectl get pods -n minio-dev
```

6.Use the kubectl port-forward command to temporarily forward traffic from the MinIO pod to the local machine
```kubectl port-forward --address=0.0.0.0 pod/minio 9000 9090```



