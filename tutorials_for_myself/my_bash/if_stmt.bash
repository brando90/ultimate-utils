#!/usr/bin/env bash

# --

if ! command -v ruby &> /dev/null
then
    ruby -v
else
    echo "no ruby"
fi

# This script will first check if the mycommand command exists using the command -v built-in. If the command is not found, the command built-in will return a non-zero exit code, which will be negated by the ! operator in the if statement. This will cause the if statement to be executed, and the echo command will be run to print "hello" to the terminal.
if ! command -v mycommand >/dev/null 2>&1; then
  # If the command does not exist, echo "hello"
  echo "hello"
fi

# --
host_v=$(hostname)
if [ $host_v = vision-submit.cs.illinois.edu ]; then
#    echo "Strings are equal."
    conda activate metalearning11.1
else
#    echo "Strings are not equal."
    conda activate metalearningpy1.7.1c10.2
fi

# --
#sudo apt-get install lsb_release
#lsb-release -a

if ! command -v ruby &> /dev/null
then
#    # First, install Ruby, as that is for some reason required to build
#    # the "system" project
#    git clone https://github.com/rbenv/ruby-build.git ~/ruby-build
#    mkdir -p ~/.local
#    PREFIX=~/.local ./ruby-build/install.sh
#    echo PREFIX
#    ~/.local/ruby-build 3.1.2 ~/.local/

    # - proverbot's version
    # First, install Ruby, as that is for some reason required to build
    # the "system" project
#    git clone https://github.com/rbenv/ruby-build.git ~/ruby-build
#    mkdir -p ~/.local
#    PREFIX=~/.local
#    echo $PREFIX
#    sh ~/ruby-build/install.sh
#    ~/.local/ruby-build 3.1.2 ~/.local/

#    # - u-pycoq's version, https://stackoverflow.com/questions/74695464/why-cant-i-install-ruby-3-1-2-in-linux
##    sudo apt-get install ruby-full
#    sudo apt-get install rbenv
#    rbenv init
#    eval "$(rbenv init - bash)"
#    echo 'eval "$(rbenv init - bash)"' >> ~/.bashrc
#
#    sudo apt-get install ruby-build
##    ruby-build 3.1.2
#    rbenv install 3.1.2
#
#    rbenv global 3.1.2
#    ruby -v

#    # - lets see if it works with a different version of ruby, crossing fingers
#    # for changing versions of ruby
#    sudo apt-get install rbenv
#    rbenv init
#    eval "$(rbenv init - bash)"
#    echo 'eval "$(rbenv init - bash)"' >> ~/.bashrc
#
#    sudo apt-get install ruby-build
##    mkdir -p ~/.local
##    PREFIX=~/.local
##    sh ~/ruby-build/install.sh
#
#    mkdir -p ~/.local
#    ruby-build 2.7.1 ~/.local
#    rbenv global 2.7.1
##    ruby-build 2.3.1p112 ~/.local/
##    ruby-build 3.1.2 ~/.local/
## rbenv install 3.1.2
##    rbenv global 3.1.2
#
#    ruby -v

    # - install rbenv locally without sudo
    mkdir ~/.rbenv
    cd ~/.rbenv
    git clone https://github.com/rbenv/rbenv.git .

    export PATH="$HOME/.rbenv/bin:$PATH"
    eval "$(rbenv init -)"
    echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc.user
    echo 'eval "$(rbenv init -)"' >> ~/.bashrc.user
#    exec $SHELL
    bash

    rbenv -v

    # - install ruby-build
    mkdir ~/.ruby-build
    cd ~/.ruby-build
    git clone https://github.com/rbenv/ruby-build.git .

    export PATH="$HOME/.ruby-build/bin:$PATH"
    echo 'export PATH="$HOME/.ruby-build/bin:$PATH"' >> ~/.bashrc.user
#    exec $SHELL
    bash

    ruby-build --version

    # - install ruby without sudo -- now that ruby build was install
    mkdir -p ~/.local
#    ruby-build 3.1.2 ~/.local/
    rbenv install 3.1.2
    rbenv global 3.1.2

    ruby -v
else
    echo "Error: failed to install ruby"
fi