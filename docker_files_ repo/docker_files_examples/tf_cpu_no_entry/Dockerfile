## This Dockerimage is meant to only yest if tensorflow can be imported when
## logged in to python3 terminal

# This means you derive your docker image from the tensorflow docker image
# FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
#FROM gcr.io/tensorflow/tensorflow
FROM ubuntu:16.04
#FROM gcr.io/tensorflow/tensorflow:latest-py3

# note this is needed to be able to install pip3
RUN apt-get update
RUN apt-get install -y vim

##RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip

# Ubuntu/Linux 64-bit, CPU only, Python 3.4
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp34-cp34m-linux_x86_64.whl
# Ubuntu/Linux 64-bit, CPU only, Python 3.5
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp35-cp35m-linux_x86_64.whl

# 3.5 r10
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

#Install some stuff my lib needs
RUN pip3 install numpy
RUN pip3 install namespaces
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install pdb
