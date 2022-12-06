#!/usr/bin/env bash

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
#
# pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
#
# pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
#
# krbtmux
# reauth
# nvidia-smi
# sh main_krbtmux.sh
#
# tmux attach -t 0

# - official install ref: https://opam.ocaml.org/doc/Install.html
# ssh brando9@hyperturing1.stanford.edu
# ssh brando9@hyperturing2.stanford.edu
# ssh brando9@turing1.stanford.edu
mkdir -p ~/.local/
mkdir -p ~/.local/bin/
bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
# type manually
~/.local/bin/
# note since it detects it in /usr/bin/opam it fails since then it tries to move opam from /usr/bin/opam to local
# ...

# if it's not at the systems level it seems to have worked
opam --versopm

# todo: without user interaction:
bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"

tr ':' '\n' <<< "$PATH"

opam init --disable-sandboxing
opam update --all
eval $(opam env)

## - opam (snap, no sudo)
## ref: https://askubuntu.com/questions/339/how-can-i-install-a-package-without-root-access
#apt-get download opam
##apt-get download opam_1.2.2-4_amd64
##ls | less
#mkdir -p ~/.local
#dpkg -x opam_1.2.2-4_amd64.deb ~/.local/bin
#export PATH="$HOME/.local/bin:$PATH"
#echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc.user
#
#tr ':' '\n' <<< "$PATH"
#
#opam --version


# - opam with conda
# maybe later, not needed I think...
# conda install -c conda-forge opam
# gave me an error in snap

# - as sudo opam
#add-apt-repository ppa:avsm/ppa
#apt update
#apt install opam


## - opam usual hack (but you should've read the opam install instructions)
#mkdir ~/.opam
#cd ~/.opam
#git clone https://github.com/rbenv/opam.git .
#
#export PATH="$HOME/.opam/bin:$PATH"
#echo 'export PATH="$HOME/.opam/bin:$PATH"' >> ~/.bashrc
#
#eval $(opam env)
#opam update --all
#opam init --disable-sandboxing
#eval $(opam env)
