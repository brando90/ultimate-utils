#!/usr/bin/env bash

# -- **consider following the instructions from an official package before the above hack**

# - install the bin then put it in path and restart your bash
mkdir ~/.rbenv
cd ~/.rbenv
git clone https://github.com/rbenv/rbenv.git .

export PATH="$HOME/.rbenv/bin:$PATH"
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
#    exec $SHELL
#bash

rbenv -v
