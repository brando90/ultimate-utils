#!/bin/sh
# this script is used to configure the container that is about to be ran.
# notice that we need the commands in a file because I am unaware of how to
# run these two commands at once in one line in a docker file. i.e.
# we need the pip3 command to install my library and we need the $@ to execute
# the incoming script command

# install my library (intention: only when the a container is spun)
pip install /home_simulation_research/hbf_tensorflow_code/my_tf_proj
# intention: do the command docker is suppose to do
$@
