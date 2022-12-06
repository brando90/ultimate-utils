#!/usr/bin/env bash

# set variable: https://askubuntu.com/questions/58814/how-do-i-add-environment-variables

# -- without export (only for current shell)
VARNAME="my value"  # DONT USE

# -- using export (to all subshells and shells)
# To set it for current shell and all processes started from current shell:
export VARNAME="my value"  # USE!

# -- using export "permanently"
# To set it permanently for all future bash sessions add such line to your .bashrc file in your $HOME directory.
# export VARNAME="my value"
# e.g. to put it right now
echo 'export VARNAME="my value"' >> ~/.bashrc