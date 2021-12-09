FROM ubuntu:16.04

LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py35 python=3.5.2 numpy pyyaml scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH
RUN conda install --name pytorch-py35 -c soumith magma-cuda80 && /opt/conda/bin/conda clean -ya
RUN conda install --name pytorch-py35 pytorch torchvision cuda80 -c soumith && /opt/conda/bin/conda clean -ya

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

#WORKDIR /workspace
#RUN chmod -R a+w /workspace

# run project depedent things
RUN conda install numpy
RUN conda install scikit-learn
RUN conda install scipy

RUN pip install matplotlib
RUN pip install sympy
RUN pip install maps
RUN pip install sklearn
RUN pip install pdb

#RUN apt install python3-tk
RUN conda install tk

RUN /bin/bash -c "source activate pytorch-py35"

# install my library (only when the a container is spun)
# bash script that configures the running container
#ENTRYPOINT ["sh", "/home_simulation_research/overparametrized_experiments/docker_files/runtime/container_setup.sh"]
#ENTRYPOINT ["sh", "/home_simulation_research/overparametrized_experiments/docker_files/runtime/fake_gui.sh"]

RUN apt-get update
RUN apt-get install -y xvfb
#RUN Xvfb :1 -screen 0 1024x768x16 &> xvfb.log  &
