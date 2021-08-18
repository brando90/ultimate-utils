#FROM ubuntu:16.04
FROM gcr.io/tensorflow/tensorflow:latest-py3
# This means you derive your docker image from the tensorflow docker image
# uses ubuntu 14.04
#FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu

# you can use apt-get to install the libraries that you need
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev
RUN apt-get install -y vim

#
#RUN apt-get install -qy python3-pip
#RUN pip install --upgrade pip
#RUN pip install --upgrade pip3
#RUN apt-get install -qy python3-pip
#RUN apt-get install -qy python3-pip --fix-missing
#RUN apt-get install -qy python3-pip
#RUN pip3 install --upgrade pip
#RUN pip install --upgrade pip

# maybe python utilities
#RUN pip install numpy scipy h5py

# you can install keras or tflearn
#RUN pip install git+https://github.com/tflearn/tflearn.git
#RUN pip install keras
