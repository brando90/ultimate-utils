# https://stackoverflow.com/a/68854067/1601580
# docker build -t miranda9_playground .
# docker run -v /Users/miranda9:/home/miranda9 -ti miranda9_playground bash
#
# doesn't work
# docker run -v /Users/miranda9:/home/miranda9 -ti continuumio/miniconda3 bash
# docker run -v /Users/miranda9:/ -ti continuumio/miniconda3 bash
#
FROM ubuntu:20.04
FROM python:latest
#FROM continuumio/anaconda3
# this one uses python 3.9
FROM continuumio/miniconda3

MAINTAINER Brando Miranda "brandojazz@gmail.com"

RUN mkdir -p /home/miranda9/
WORKDIR /home/miranda9

# this wont work, read this some day when I need to activate my env in docker: https://pythonspeed.com/articles/activate-conda-dockerfile/
#RUN conda create -n docker_env python=3.9 -y
#RUN conda init bash
#RUN conda activate docker_env

#RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#    ssh \
#    git \
#    opam \
#    wget \
#    ca-certificates
#
#RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh \
#  && chmod +x Anaconda3-2020.02-Linux-x86_64.sh \
#  && bash Anaconda3-2020.02-Linux-x86_64.sh -b -f
#ENV PATH="/root/anaconda3/bin:${PATH}"
#RUN conda create -n synthesis python=3.9 -y
#ENV PATH="/root/anaconda3/envs/synthesis/bin:${PATH}"