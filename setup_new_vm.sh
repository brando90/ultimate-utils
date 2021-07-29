apt update;
apt install build-essential;

sudo add-apt-repository ppa:graphics-drivers
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt-get install nvidia-driver-450
# sudo apt-get --purge remove nvidia-driver-460
reboot now

adduser miranda9
sudo adduser miranda9 sudo

#
apt-get install opam

# get conda
miniconda_linux=Miniconda3-latest-Linux-x86_64.sh
# miniconda_darwin=Miniconda3-latest-MacOSX-x86_64.sh
miniconda_host=https://repo.anaconda.com/miniconda/
curl -O $miniconda_host$miniconda_linux
sh $miniconda_linux -b -f
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda initeval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init

conda install -y conda-build
conda update -n base -c defaults conda

#
git clone git@github.com:brando90/ultimate-utils.git

#
conda create -n synthesis python=3.9
conda activate synthesis


#
#sudo su -
#apt-get install -y ubuntu-drivers-common
#ubuntu-drivers autoinstall
#apt install -y -q nvidia-driver-460
#nvidia-smi

# You might want to try this first ( from: https://www.itzgeek.com/post/how-to-install-nvidia-drivers-on-ubuntu-20-04-ubuntu-18-04.html)
#	sudo apt install -y nvidia-driver-460-server
#If there's a problem…
#make sure the kernel headers are installed:
#	sudo apt install linux-headers-$(uname -r)
#remove and purge driver
#	sudo apt-get remove -y  --purge nvidia* && sudo apt autoremove
#	sudo apt -y update
#	sudo apt -y upgrade
#	sudo apt -y dist-upgrade
#	sudo apt -y autoremove
#	sudo apt install -y nvidia-driver-460-server
#	sudo reboot
#	…
#	nvidia-smi
#Other  methods
#Nvidia driver installation for Ubuntu 18.04 (2020)
#https://medium.com/@sreenithyc21/nvidia-driver-installation-for-ubuntu-18-04-2020-2918be830d0f
#Update/upgrade ubuntu:
#sudo apt update
#sudo apt upgrade
#sudo apt dist-upgrade
#sudo apt autoremove
#sudo reboot
#nvidia-smi
#If nvidia-smi doesn't work
#sudo apt list --installed | fgrep nvidia
#sudo apt-get remove --purge '^nvidia-.*'
#sudo apt-get --purge remove libnvidia-*
#sudo apt-get remove --purge 'cuda-.*'
#sudo apt autoremove
#sudo ppa-purge  ppa:graphics-drivers/ppa   # using spa lib means no nvidia-smi
#sudo rm -i /etc/apt/sources.list.d/graphics-drivers-ubuntu-ppa-bionic.list*
## to prevent "Re: NVIDIA driver install - Error: Unable to find the kernel source tree" in installation step
#sudo apt-get install linux-headers-`uname -r`
## get the nvidia driver from nvidia.com; make sure you drill down and get the version for ubuntu 18.04- generic linux won't work
#sudo Downloads/NVIDIA-Linux-x86_64-460.32.03.run
#To Lock the nvidia code (Adapted from:https://chrisalbon.com/deep_learning/setup/prevent_nvidia_drivers_from_upgrading/)
#sudo apt-mark hold nvidia-compute-utils-460 nvidia-dkms-460 nvidia-driver-460 nvidia-kernel-common-460 nvidia-kernel-source-460 nvidia-modprobe nvidia-prime nvidia-settings nvidia-utils-460
#Locking the version will stop the security updates, but you'll see security warnings quite soon.

#---
#  - name: Update everything
#    apt:
#      upgrade: safe
#      update_cache: yes
#
#  - name: Install dependencies
#    apt: name=python3-pip
#
#  - name: Add an Apt signing key, will not download if present
#    apt_key:
#      id: 7fa2af80
#      url: http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
#      state: present
#
#  - name: Check for nvidia-smi
#    stat: path=/usr/bin/nvidia-smi
#    register: st
#
#  - name: Install CUDA References
#    apt:
#      deb: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
#      state: present
#    when: not st.stat.exists
#
#  - name: Install cuda
#    apt:
#      name: cuda-10-0
#      update_cache: yes
#
#  - include_role:
#      name: anaconda
#
#  - name: install deep learning packages via conda
#    conda:
#      name: "{{ item }}"
#      state: latest
#    with_items:
#      - numpy
#      - tensorflow-gpu
#      - matplotlib
#      - jupyter
#      - keras
#      - scikit-learn
#      - scipy


#apt update &&
#apt upgrade -y &&
#apt install build-essential -y &&
#wget https://us.download.nvidia.com/tesla/460.73.01/NVIDIA-Linux-x86_64-460.73.01.run &&
#chmod +x NVIDIA-Linux-x86_64-460.73.01.run &&
#./NVIDIA-Linux-x86_64-460.73.01.run &&
#shutdown -r now;