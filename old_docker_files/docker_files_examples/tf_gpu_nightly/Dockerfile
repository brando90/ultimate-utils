#FROM ubuntu:16.04
# This means you derive your docker image from the tensorflow docker image
FROM tensorflow/tensorflow:nightly-gpu-py3

# note this is needed to be able to install pip3
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev
RUN apt-get install -y vim

# get python 3
#RUN apt-get install -y python3
#RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

#Install some stuff my lib needs
#RUN pip3 install tensorflow-gpu
RUN pip install numpy
RUN pip install namespaces
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install pdb
RUN pip install keras
RUN pip install pandas

# install my library (only when the a container is spun)
# bash script that configures the running container
ENTRYPOINT ["sh", "/home_simulation_research/hbf_tensorflow_code/docker_files/tf_gpu/container_setup.sh"]

# enviromnet
ENV OMP_NUM_THREADS=2
