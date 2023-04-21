# Local Deployment

https://min.io/docs/minio/macos/index.html

For arm42:
#### MNIO:

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
Run the local server from the address: http://127.0.0.1:9000 and log in with credintials
(MinIO automatically redirects browser access to the MinIO Console)


Now you can create buckets, upload your files there and create service accounts

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

# KUBERNETES

https://min.io/docs/minio/kubernetes/upstream/index.html

1. Download yaml with kubernetes setting
```angular2html
curl https://raw.githubusercontent.com/minio/docs/master/source/extra/examples/minio-dev.yaml -O
```
2. Create cluster 
```
kind create cluster --name ml-in-production-course-week-1
```
3. Apply yaml
```angular2html
kubectl apply -f minio-dev.yaml
--------------
namespace/minio-dev created
pod/minio created
```
4. Verify the state
```angular2html
kubectl get pods -n minio-dev
```

5. Use the kubectl port-forward command to temporarily forward traffic from the MinIO pod to the local machine


