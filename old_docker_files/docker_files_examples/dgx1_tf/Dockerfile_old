# This means you derive your docker image from the tensorflow docker image
FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu

# you can use apt-get to install the libraries that you need
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev

# maybe python utilities
RUN pip install numpy scipy h5py

# you can install keras or tflearn
RUN pip install git+https://github.com/tflearn/tflearn.git
RUN pip install keras
