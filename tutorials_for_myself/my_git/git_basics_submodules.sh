# decided against this because it seems complicated

# - note to clone uutils with its submodule do (cmd not tested):
cd $HOME
git clone --recurse-submodules git@github.com:brando90/ultimate-utils.git

# - git submodules
cd $HOME/diversity-for-predictive-success-of-meta-learning

# - in case it's needed if the submodules bellow have branches your local project doesn't know about from the submodules upstream
git fetch

# -- first repo
# - adds the repo to the .gitmodule & clones the repo
git submodule add -f -b hdb --name meta-dataset git@github.com:brando90/meta-dataset.git meta-dataset/
git submodule add -f -b hdb --name pytorch-meta-dataset git@github.com:brando90/pytorch-meta-dataset.git pytorch-meta-dataset/

# - ref for init then update: https://stackoverflow.com/questions/3796927/how-do-i-git-clone-a-repo-including-its-submodules/3796947#3796947
#git submodule init
#git submodule update

# - git submodule init initializes your local configuration file to track the submodules your repository uses, it just sets up the configuration so that you can use the git submodule update command to clone and update the submodules.
git submodule init
# - git submodule update --init initializes your local configuration file and clones the submodules for you, using the commit specified in the main repository.
#   note, command bellow will not pull the right branch -- even if it's in your .gitmodules file, for that you need remote. Likely because it looks at the origin (pointer to remote) in github for the available branches.
#   note, bellow pulls the submodules if you didn't specify them when cloning parent project, ref: https://youtu.be/wTGIDDg0tK8?t=119
git submodule update --init
# - The --remote option tells Git to update the submodule to the commit specified in the upstream repository, rather than the commit specified in the main repository. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
#git submodule update --init --remote
git submodule update --init --recursive --remote meta-dataset

# - check we are using the right branch https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
git submodule status
cd meta-dataset
git branch
cd ..

# - for each submodule pull from the right branch according to .gitmodule file. ref: https://stackoverflow.com/questions/74988223/why-do-i-need-to-add-the-remote-to-gits-submodule-when-i-specify-the-branch?noredirect=1&lq=1
#git submodule foreach -q --recursive 'git switch $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master || echo main )'

# - check it's in specified branch ref: https://stackoverflow.com/questions/74998463/why-does-git-submodule-status-not-match-the-output-of-git-branch-of-my-submodule
git submodule status
cd meta-dataset
git branch
cd ..

# - pip install
#pip install -e -r meta-dataset/requirements.txt
#pip install -e -r pytorch-meta-dataset/requirements.txt

# - clone & pull git submodules
# git clone --recurse-submodules --remote-submodules <repo-URL>
