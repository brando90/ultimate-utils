# My Docker Tutorial

Note, if start the "getting-started" application as a docker container you can see a neat tutorial by
docker at http://localhost/tutorial/.
If you have docker desktop try running: 
`docker run -d -p 80:80 --name docker-tutorial docker101tutorial`

General tip:
you can do:
```
man docker ps
docker ps -h
docker ps --help
```
to see information about each command, in the above the command `docker ps`.
Man usually opens a much more extensive description.

## High level idea of docker workflow

Usual workflow:
0. way to create image (e.g. `Dockerfile`)
1. building the image
2. run the container/application from the image with a docker container
3. sharing/pushing the image to your hub

## Example from the high lever idea 

0. go to the section on how to build `Dockerfile`s.

1. build an image from a docker file `Dockerfile`
```
docker build -t getting-started .
```
builds a docker image tagged (`-t`) `getting-started`. 
Now you can start run the application from the docker image as a docker container.

2. run the application as a docker container from the docker image
```
docker run -d -p 80:80 docker/getting-started
```
this runs an application as a docker container from the docker image `docker/getting-started`.
The `-d` is the detached mode (runs in background).
The `-p` map port 80 of the host to port 80 in the container.
Note you can combine single character flags as follow: `-dp`.
Note you can see your docker containers in the docker desktop or with
`docker ps -a`. The flag `-a` is for all.

## Definitions

image = When running a container, it uses an isolated filesystem. This custom filesystem is provided by a container image. 
Since the image contains the container's filesystem, it must contain everything needed to run an application - all 
dependencies, configuration, scripts, binaries, etc. The image also contains other configuration for the container, 
such as environment variables, a default command to run, and other metadata.
We'll dive deeper into images later on, covering topics such as layering, best practices, and more.

container = Now that you've run a container, what is a container? 
Simply put, a container is simply another process on your machine that has been isolated from all other processes on the host machine. That isolation leverages kernel namespaces and cgroups, features that have been in Linux for a long time. 
Docker has worked to make these capabilities approachable and easy to use.

### Questions

Questions:
- what exactly is an image? does it have my data, code, filesystem? or does it just save the instructions - if yes how 
different from a Dockerfile?
- how does it have access to my source code? is it different to how it has access to data?
- does the saved image save the files/source code I need or do I have to gitclone it every time?
- docker `docker ps` vs docker dashboard
- 

### Running docker getting started tutorial

Docker getting started:

1. todo, what do I need to build an image? e.g. Docker file. A Dockerfile is simply a text-based script of instructions that is used to create a container image.
```
    TODO get cmd
```

2. build an image: A Docker image is a private file system just for your container. It provides all the files and code your container needs.
```
    docker run -d -p 80:80 --name docker-tutorial docker101tutorial
```
3. run container: 
```
    docker run -d -p 80:80 --name docker-tutorial docker101tutorial
```

4. push image: Now save and share your image. Save and share your image on Docker Hub to enable other users to easily download and run the image on any destination machine.
```
    docker tag docker101tutorial brandojazz/docker101tutorial
	docker run -d -p 80:80 --name docker-tutorial docker101tutorial
 ```
then go here to see the getting started tutorial that your running in docker: 
http://localhost/tutorial/  