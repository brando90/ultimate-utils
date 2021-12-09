# https://hub.docker.com/r/continuumio/anaconda3/
FROM continuumio/anaconda3

# note this is needed to be able to install pip3
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev
RUN apt-get install -y vim

RUN pip install --upgrade pip

#Install some stuff my lib needs
RUN pip install tensorflow-gpu
RUN pip install numpy
RUN pip install namespaces
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install pdb
RUN pip install keras

# install my library (only when the a container is spun)
# bash script that configures the running container
ENTRYPOINT ["sh", "/home_simulation_research/hbf_tensorflow_code/docker_files/tf_gpu_anaconda/container_setup.sh"]

# enviromnet
ENV OMP_NUM_THREADS=2
