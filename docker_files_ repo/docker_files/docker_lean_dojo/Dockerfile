# # Use a specific version of kitware/cmake as the base image
# FROM kitware/cmake:ci-clang_cxx_modules-x86_64-2023-02-15

# # Install required packages: 'which', GMP library development files, Python3, and pip
# RUN yum -y install which gmp-devel python3 python3-pip
# # Create a symbolic link for python3 as /usr/bin/python for compatibility
# RUN ln -s $(which python3) /usr/bin/python
# # Install Python packages: toml for TOML file parsing, loguru for logging, tqdm for progress bars
# RUN pip3 install toml loguru tqdm

# # Set the environment variable ELAN_HOME to store Elan's installation
# ENV ELAN_HOME="/.elan"
# # Add ELAN_HOME/bin to the PATH for easier access to Elan's binaries
# ENV PATH="${ELAN_HOME}/bin:${PATH}"
# # Download and execute Elan's installation script in silent mode with auto-approval
# RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y

# # Change permissions to allow writing in the /.elan directory
# RUN chmod -R a+w /.elan

# # Set the working directory to /workspace for the container
# WORKDIR /workspace
# # Change permissions to allow writing in the /workspace directory
# RUN chmod -R a+w /workspace

FROM ubuntu:16.04
RUN echo "hello"
