FROM continuumio/miniconda3

# bash > sh ...
SHELL ["/bin/bash", "--login", "-c"]

# minimal set of utilities
# Run the following as root:
RUN apt-get update -y -q \
 && apt-get install -y -q --no-install-recommends \
    # alphabetical order advised for long package lists to ease review & update
    ca-certificates \
    curl \
    wget \
    git \
    libgmp-dev \
    m4 \
    opam \
    rsync \
    sudo \
    python3.9 \
    vim \
# Docker-specific cleanup to earn space
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# create conda env
RUN conda init bash
RUN conda create -n pycoq python=3.9
RUN echo "conda activate pycoq" >> ~/.bashrc
ENV PATH /opt/conda/pycoq/env/bin:$PATH

# opam is the easiest way to install coqc
RUN opam init --auto-setup --yes --bare --disable-sandboxing 
# && opam pin add -y -k version -j "$(nproc)" coq 8.14.0

# start setting up pycoq
# first create a switch for our project
RUN opam switch create pycoq-ejgallego-coq-8.14 4.12.0
RUN eval $(opam env --switch=pycoq-ejgallego-coq-8.14 --set-switch)

# Clone pycoq-emilio repo cuz he says so (this shouldn't be something a python user needs to do)
RUN mkdir /workspaces
WORKDIR /workspaces
RUN git clone https://github.com/brando90/pycoq-ejgallego.git
WORKDIR ./pycoq-ejgallego
RUN git submodule update --init --recursive

# pin essential package to the correct versions
RUN opam pin add -y coq 8.14.0
RUN opam install -y --deps-only coq-serapi
RUN opam pin add pythonlib v0.14.0 --yes
RUN opam install -y pythonlib
RUN opam install -y --deps-only .

# pin version of cmdliner and pyml to older versions, key to successfully running pycoq in year 2023.
RUN opam pin add cmdliner 1.0.4 --yes
RUN opam pin add pyml 20211015 --yes

# Docker-specific cleanup to earn space
RUN opam clean -a -c -s --logs

# automate eval opam env
ENTRYPOINT ["opam", "exec", "--"]
CMD ["/bin/bash", "--login"]



