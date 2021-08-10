# install pytorch
#conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge
#conda install -y pytorch torchvision torchaudio -c pytorch -c conda-forge

# for graphviz, todo check which os https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux
# sudo port install graphviz
brew install graphviz
# sudo apt install graphviz


conda install -y dill
conda install -y networkx">=2.5"
conda install -y scipy
conda install -y scikit-learn
conda install -y lark-parser -c conda-forge
conda install -y -c pytorch torchtext
conda install -y -c conda-forge tensorboard
conda install -y -c conda-forge pandas
conda install -y -c conda-forge progressbar2

conda install -y pygraphviz -c conda-forge

# similar to pip -e
conda install conda-build
#conda develop ~/ultimate-utils/ultimate-utils-proj-src
