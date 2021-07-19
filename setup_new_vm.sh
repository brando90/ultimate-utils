apt update;
apt install build-essential;

sudo add-apt-repository ppa:graphics-drivers
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
sudo apt-get install nvidia-driver-450
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