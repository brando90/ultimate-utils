#!/usr/bin/env bash

# - get ubuntu version for ubuntu img
#docker run -it --rm ubuntu:20.04 /bin/bash
docker run -it --rm ubuntu:18.04 /bin/bash
apt-get update && apt-get install -y lsb-release && apt-get clean all
lsb_release -a
#apt-get install git

apt-get update
apt-get upgrade ruby-build

apt-get install rbenv
apt-get install ruby-build
rbenv install 3.1.2

# - get ubuntu version for miniconda3 docker img
docker run -it --rm continuumio/miniconda3:latest /bin/bash
#sudo apt-get install lsb-release
#apt-get install lsb-release
apt-get update && apt-get install -y lsb-release && apt-get clean all
lsb_release -a


# - 4 commands to try
lsb_release -a
lsb_release -d command line
cat /etc/lsb-release
cat /etc/issue c