docker build -t resnet_model -f api/Dockerfile api/
docker run -d -p 5152:8080 --name resnet_container resnet_model

docker login
docker tag resnet_model mlopsteam4/resnet_model:latest
docker push mlopsteam4/resnet_model:latest