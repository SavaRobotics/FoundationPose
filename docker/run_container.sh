#!/usr/bin/env bash

# kill any old one
docker rm -f foundationpose 2>/dev/null || true

# base folder for volume mounts
DIR=$(pwd)/../

docker run \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  --network host \
  --name foundationpose \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $DIR:$DIR \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  foundationpose:latest \
  bash -c "cd $DIR && bash"

bash build_all.sh

mkdir -p groundingdino/weights/
wget -P groundingdino/weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth