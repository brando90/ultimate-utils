# Docker for Lean read me

Borrowed from [leandojo](https://github.com/lean-dojo/LeanDojo/blob/main/docker/build_docker_image.sh)
```bash
#! /usr/bin/sh
# This is for developers only. Users do not have to build Docker containers.
# To use LeanDojo with Docker, users just have to make sure Docker is installed and running.
cd $HOME/gold-ai-olympiad/docker
docker buildx build --platform linux/amd64,linux/arm64 -t brandojazz/lean-dojo:latest --push .
```
If multiplataform docker image creation fails hopefully Kaiyu helps us fix https://github.com/lean-dojo/LeanDojo/issues/137 .

If buildx doesn't work then create the docker image for your plataform:
```bash
cd $HOME/gold-ai-olympiad/docker

# Build the Image
docker build -t brandojazz/lean-dojo:latest .

# Push the Image to a Docker Registry
docker push brandojazz/lean-dojo:latest

# Start the Docker Container, -d, it runs the container in the background 
docker run -d --name lean-dojo-container brandojazz/lean-dojo:latest
# docker run -it --name lean-dojo-container brandojazz/lean-dojo:latest bash

# Attach to the running Docker container via a terminal CLI bash
docker exec -it lean-dojo-container bash
```

## Reminder of Docker concepts

Q: remind me difference between containers and images one sentence for diff?
- A Docker image is a lightweight, standalone, and immutable package that includes everything needed to run a piece of software, including the code, runtime, libraries, and environment variables. 
- A Docker container is a runtime instance of a Docker image, which runs in isolation but can communicate with other containers and the host system. 
- The key difference is that an image is a static blueprint, while a container is a live, running instance of that blueprint.


## Standard way to build docker images and containers

```bash
# Build an Image from the Dockerfile:
docker build -t <image_name> .
# Run a Container from the Image:
docker run -d <image_name>

```


## Refs
ref:
    - multiplatafrom build fails: https://github.com/lean-dojo/LeanDojo/issues/137 
    - multiplatafrom issue SO: https://stackoverflow.com/questions/78054282/how-to-build-the-multiplataform-docker-image-on-macos 
    - kitware arm: https://stackoverflow.com/questions/78054300/what-is-a-good-base-image-for-arm-for-the-mac-os-x-instead-of-from-kitware-cmake