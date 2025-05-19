#!/bin/bash

# 镜像名和容器名（可根据需要修改）
IMAGE_NAME=ewtsvd_app_image
CONTAINER_NAME=ewtsvd_app_container
HOST_PORT=5301
CONTAINER_PORT=5300

echo "正在清理旧容器（如果有）..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null
docker rmi $IMAGE_NAME 2>/dev/null 

echo "正在构建 Docker 镜像..."
docker build -t $IMAGE_NAME .

echo "正在运行 Docker 容器..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $HOST_PORT:$CONTAINER_PORT \
  -v $(pwd)/uploads:/app/uploads \
  $IMAGE_NAME

echo "部署完成！容器 $CONTAINER_NAME 正在运行，访问端口为 $HOST_PORT。"

