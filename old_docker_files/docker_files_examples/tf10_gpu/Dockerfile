# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
#FROM gcr.io/tensorflow/tensorflow
#FROM ubuntu:16.04
FROM gcr.io/tensorflow/tensorflow:latest-py3

# note this is needed to be able to install pip3
RUN apt-get update
RUN apt-get install -y build-essential git
RUN apt-get install -y vim

##RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip

# py 3.5 r10 gpu ubuntu
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

#Install some stuff my lib needs
RUN pip3 install numpy
RUN pip3 install namespaces
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install pdb

# install my library (only when the a container is spun)
# bash script that configures running container
ENTRYPOINT ["sh", "/home_simulation_research/hbf_tensorflow_code/docker_files/tf_cpu/container_setup.sh"]

# enviromnet
ENV OMP_NUM_THREADS=2
