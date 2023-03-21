# Playging with PyCoq-Ejgallego 

## Getting Started

```bash
# --- To be safe lets create a switch only for pycoq-emilio
# -- to build PyCoq, you need Python 3 and an OCaml environment able to build SerAPI; usually, this can be achieved using the OPAM package manager and doing:
# -- for the 0.13 version that targets Coq v8.13; note that OCaml >= 4.11.0 is recommended, and >= 4.08.0 required by the dependencies.
opam switch create pycoq-ejgallego-coq-8.14 4.12.0
eval $(opam env --switch=pycoq-ejgallego-coq-8.14 --set-switch)
opam pin add -y coq 8.14.0
# Install the packages that can be installed directly through opam
opam repo add coq-released https://coq.inria.fr/opam/released
opam repo add coq-extra-dev https://coq.inria.fr/opam/extra-dev

# --- Create conda env
#conda create -n iit_synthesis python=3.9
#conda activate iit_synthesis
conda create -n pycoq-ejgallego python=3.9
conda activate pycoq-ejgallego
#pip install -e ~/ultimate-utils

# - Clone pycoq-emilio repo cuz he says so (this shouldn't be something a python user needs to do)
cd ~
git clone git@github.com:brando90/pycoq-ejgallego.git
cd ~/pycoq-ejgallego
# pip install -e .
git submodule update --init --recursive

# --- To build PyCoq, you need Python 3 and an OCaml environment able to build SerAPI; usually, this can be achieved using the OPAM package manager and doing:
opam install --deps-only coq-serapi
#opam install -y coq-serapi
opam pin add pythonlib v0.14.0
opam install -y pythonlib
cd ~/pycoq-ejgallego
opam install --deps-only .

# pin version of cmdliner and pyml to older versions, key to successfully running pycoq in year 2023.
opam pin add cmdliner 1.0.4
opam pin add pyml 20211015

#- build pycoq-egjgallego
make install && dune build examples/test.py && dune exec -- python3 _build/default/examples/test.py

# --- If you want an interactive environment, use:
make install && dune exec -- python

import os
os.chdir('_build/default')
import pycoq, coq
```

