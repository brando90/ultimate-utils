#!/usr/bin/env bash

# - install the bin then put it in path and restart your bash
mkdir ~/.rbenv
cd ~/.rbenv
git clone https://github.com/rbenv/rbenv.git .

echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
#    exec $SHELL
bash

rbenv -v

# - opam (usual local hack)
#sudo apt-get install m4 curl wget unzip git
#
#mkdir ~/.opam
#cd ~/.opam
#git clone https://github.com/ocaml/opam.git .
#
#sh ./config
#make
#make install
#
#echo 'export PATH="$HOME/.opam/bin:$PATH"' >> ~/.bashrc.user
#export PATH="$HOME/.opam/bin:$PATH"
#
##exec $SHELL
##bash
#exit
#ssh brando9@hyperturing1.stanford.edu
#
#$ opam --version

# - opam (snap, no sudo)
apt-get download opam
#apt-get download opam_1.2.2-4_amd64
#ls | less
mkdir -p ~/.local
dpkg -x opam_1.2.2-4_amd64.deb ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc.user

tr ':' '\n' <<< "$PATH"

#apt-get update && apt-get install -y lsb-release && apt-get clean all && apt-get install -y curl
#mkdir ~/.bubblewrap
#cd ~/.bubblewrap
#git clone https://github.com/containers/bubblewrap.git .
#
#sh ./autogen.sh
#sh ./configure
#make
#make install
#
#sudo setcap cap_sys_admin+ep $(which bubblewrap)
#
#
#mkdir -p ~/.local/
#bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
## I manually typed the local were I do have permissions:
#~/.local/
#
#export PATH="$HOME/.bubblewrap/bin:$PATH"
#echo 'export PATH="$HOME/.bubblewrap/bin:$PATH"' >> ~/.bashrc.user
##exec $SHELL
##bash

apt-get install bubblewrap -t ~/.local/

bubblewrap --version


opam --version

# - this one is not **essential** because the above one worked and non interactive works with opam in the terminal
# idk if this one works, anyway to late here is the SO post: https://stackoverflow.com/questions/74696218/how-does-one-install-opam-without-sudo-in-ubuntu-without-interaction-from-the-us
#mkdir -p ~/.local/
#echo "~/.local/" | bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"

# - opam with conda
# maybe later, not needed I think...
# conda install -c conda-forge opam
# gave me an error in snap

# - as sudo opam
#add-apt-repository ppa:avsm/ppa
#apt update
#apt install opam
