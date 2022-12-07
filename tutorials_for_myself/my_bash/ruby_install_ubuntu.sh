#!/usr/bin/env bash

# -- apt isn't recommended, always see official install isntructions
#RUN apt-get install -y --no-install-recommends rbenv
#RUN apt-get install -y --no-install-recommends ruby-build
#RUN apt-get install -y --no-install-recommends ruby-full
#RUN rbenv install 3.1.2
#RUN rbenv global 3.1.2

# -- apt isn't recommended, always see official install isntructions
#sudo apt-get install -y --no-install-recommends rbenv
#sudo apt-get install -y --no-install-recommends ruby-build
#sudo apt-get install -y --no-install-recommends ruby-full
#rbenv install 3.1.2
#rbenv global 3.1.2

# -- apt isn't recommended, always see official install isntructions
#sudo apt install -y build-essential bison zlib1g-dev libyaml-dev libssl-dev libgdbm-dev libreadline-dev libffi-dev
#wget https://cache.ruby-lang.org/pub/ruby/3.1/ruby-3.1.2.tar.xz
#tar -xJvf ruby-3.1.2.tar.xz
#cd ruby-3.1.2
#./configure --prefix=$HOME/.rbenv/versions/3.1.2
#make -j2
#sudo make install
#
#rbenv install 3.1.2
#rbenv global 3.1.2
#
#ruby --version

# --
#I don't think there's an apt package. I clone the repo and run sudo make install or these are download instructions: https://github.com/postmodern/ruby-install#install
# If you clone the repo you can get pull and run sudo make install every few years. 😛
  #The tool can update Ruby definitions independently so no need to update frequently.
  #Or you can build Ruby manually for rbenv with a bit of additional fuss too. I'd be happy to help if you want to go that route.
  #It usually works to just quickly install ruby-install. It's very little code and you can tear it back down without much ado.

# -- failed attempt documented here: https://stackoverflow.com/questions/74695464/why-cant-i-install-ruby-3-1-2-in-linux-docker-container?noredirect=1#comment131843536_74695464
# suggestion from discord ruby
#git clone https://github.com/rbenv/ruby-build.git "$(rbenv root)"/plugins/ruby-build
#git -C "$(rbenv root)"/plugins/ruby-build pull
#rbenv install 3.1.2
#rbenv global 3.1.2

# ---- official instructions: https://github.com/rbenv/rbenv#basic-git-checkout
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
echo 'eval "$(~/.rbenv/bin/rbenv init - bash)"' >> ~/.bashrc
eval "$(~/.rbenv/bin/rbenv init - bash)"
bash
exec $SHELL
source ~/.bashrc
# Restart your shell so that these changes take effect. (Opening a new terminal tab will usually do it.)

rbenv install 3.1.2
rbenv global 3.1.2

ruby --version
which ruby

# -- bellow worked on snap cluster but not on docker container
# - install rbenv for installing ruby
mkdir ~/.rbenv
cd ~/.rbenv
git clone https://github.com/rbenv/rbenv.git .

export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)"
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc.user
echo 'eval "$(rbenv init -)"' >> ~/.bashrc.user
exec $SHELL
bash
source ~/.bashrc.user

rbenv -v

# - install ruby-build
mkdir ~/.ruby-build
cd ~/.ruby-build
git clone https://github.com/rbenv/ruby-build.git .

export PATH="$HOME/.ruby-build/bin:$PATH"
echo 'export PATH="$HOME/.ruby-build/bin:$PATH"' >> ~/.bashrc.user
exec $SHELL
bash
source ~/.bashrc.user

ruby-build --version

# - install ruby without sudo -- now that ruby build was install
mkdir -p ~/.local
#    ruby-build 3.1.2 ~/.local/
rbenv install 3.1.2
rbenv global 3.1.2

ruby -v
which ruby