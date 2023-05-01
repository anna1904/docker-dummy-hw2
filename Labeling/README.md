
1. Run Docker image with label studio inside
```angular2html
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest
```
2. Open the `http://localhost:8080` and signup/login to the Label studio
3. Load 50 samples of images
4. Write labels for categories and label data