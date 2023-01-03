# updating all git submodules according to branch in .gitmodule file in super proj
# https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch
# https://git-scm.com/docs/git-submodule/#Documentation/git-submodule.txt-foreach--recursiveltcommandgt

# -- pretend you've add the submodules so far
git submodule add -f -b hdb --name meta-dataset git@github.com:brando90/meta-dataset.git meta-dataset/
git submodule add -f -b hdb --name pytorch-meta-dataset git@github.com:brando90/pytorch-meta-dataset.git pytorch-meta-dataset/

# - init local config & try to pull (from remote/branch or initializes your local configuration file and clones the submodules for you, using the commit specified in the main repository.)
#   ref: https://youtu.be/wTGIDDg0tK8?t=119, https://stackoverflow.com/questions/44366417/what-is-the-point-of-git-submodule-init
git submodule init
git submodule update --init
#git submodule update --init --recursive --remote

git submodule status

# - for each submodule pull from the right branch according to .gitmodule file
# ref: doc for "foreach" cmd: https://git-scm.com/docs/git-submodule/#Documentation/git-submodule.txt-foreach--recursiveltcommandgt
# ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch#74994315
# note: The command has access to the variables $name, $sm_path, $displaypath, $sha1 and $toplevel...
# note: $toplevel is: $toplevel is the absolute path to the top-level of the immediate superproject.
# note: execute a command in a subshell $(...) ($(command) is known as command substitution. It allows the output of a command to be used as an argument to another command. )
# note: get the submodule.$name.branch of the current ($name) submodule, as visited by the git submodule foreach command.
git submodule foreach -q --recursive \
  'git switch \
  $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master || echo main )'

# - check status of one of the submodules for unit test above worked: https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
# note: in case response bellow says origin: "origin" typically refers to a remote repository that is associated with your local repository.
git submodule status
cd meta-dataset
git branch  # should show hdb
cd ..

