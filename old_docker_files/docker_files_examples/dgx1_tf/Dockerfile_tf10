FROM ubuntu:16.04
# you can use apt-get to install the libraries that you need
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev

#
RUN apt-get install -qy python3-pip
RUN pip3 install --upgrade pip
RUN pip install --upgrade pip
RUN apt-get install -qy python3-pip --fix-missing

# maybe python utilities
RUN pip install numpy scipy h5py

# you can install keras or tflearn
RUN pip install git+https://github.com/tflearn/tflearn.git
RUN pip install keras

# py 3.5 r10 gpu ubuntu
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
