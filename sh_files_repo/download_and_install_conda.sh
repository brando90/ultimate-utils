# RUN ALL THE INSTRUCTIONS! PLEASE!
echo $HOME
cd $HOME
# -- Install miniconda
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
################################################################################################
   '''You should get something like  
  「--2023-10-31 00:17:21--  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  　Resolving repo.anaconda.com (repo.anaconda.com)... 104.16.131.3, 104.16.130.3
　　Connecting to repo.anaconda.com (repo.anaconda.com)|104.16.131.3|:443... connected.      
　　HTTP request sent, awaiting response... 200 OK
　　Length: 120771089 (115M) [application/x-sh]
　　Saving to: ‘/lfs/mercury1/0/mariodp/miniconda.sh’

　　/lfs/mercury1/0/mariod 100%[=========================>] 115.18M  85.2MB/s    in 1.4s    

　　2023-10-31 00:17:22 (85.2 MB/s) - ‘/lfs/mercury1/0/mariodp/miniconda.sh’ saved [120771089/120771089]」'''
################################################################################################

#Then, do 
bash $HOME/miniconda.sh -b -p $HOME/miniconda
################################################################################################
　　'''You should get something like:
  　　「PREFIX=/lfs/mercury1/0/mariodp/miniconda
    　　Unpacking payload ...
    　　Installing base environment...

    　　Downloading and Extracting Packages


    　　Downloading and Extracting Packages

    　　Preparing transaction: done
    　　Executing transaction: done
    　　installation finished.
    　　mariodp@mercury1:~$ source $HOME/miniconda/bin/activate
    　　(base) mariodp@mercury1:~$ conda init
    　　no change     /lfs/mercury1/0/mariodp/miniconda/condabin/conda
    　　no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda
    　　no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda-env
    　　no change     /lfs/mercury1/0/mariodp/miniconda/bin/activate
    　　no change     /lfs/mercury1/0/mariodp/miniconda/bin/deactivate
    　　no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.sh
    　　no change     /lfs/mercury1/0/mariodp/miniconda/etc/fish/conf.d/conda.fish
    　　no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/Conda.psm1
    　　no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/conda-hook.ps1
    　　no change     /lfs/mercury1/0/mariodp/miniconda/lib/python3.11/site-packages/xontrib/conda.xsh
    　　no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.csh
    　　modified      /lfs/mercury1/0/mariodp/.bashrc

    　　==> For changes to take effect, close and re-open your current shell. <==」'''
################################################################################################ 

#next, type
source $HOME/miniconda/bin/activate
# - Set up conda
conda init
#############################################################################################
    '''You'll see something like
    「no change     /lfs/mercury1/0/mariodp/miniconda/condabin/conda
    no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda
    no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda-env
    no change     /lfs/mercury1/0/mariodp/miniconda/bin/activate
    no change     /lfs/mercury1/0/mariodp/miniconda/bin/deactivate
    no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.sh
    no change     /lfs/mercury1/0/mariodp/miniconda/etc/fish/conf.d/conda.fish
    no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/Conda.psm1
    no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/conda-hook.ps1
    no change     /lfs/mercury1/0/mariodp/miniconda/lib/python3.11/site-packages/xontrib/conda.xsh
    no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.csh
    no change     /lfs/mercury1/0/mariodp/.bashrc
    No action taken.」'''
################################################################################################

# conda init zsh
conda init bash
#############################################################################################
    '''You'll see something like
    「no change     /lfs/mercury1/0/mariodp/miniconda/condabin/conda
    　no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda
    　no change     /lfs/mercury1/0/mariodp/miniconda/bin/conda-env
　    no change     /lfs/mercury1/0/mariodp/miniconda/bin/activate
　    no change     /lfs/mercury1/0/mariodp/miniconda/bin/deactivate
　    no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.sh
    　no change     /lfs/mercury1/0/mariodp/miniconda/etc/fish/conf.d/conda.fish
　    no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/Conda.psm1
　    no change     /lfs/mercury1/0/mariodp/miniconda/shell/condabin/conda-hook.ps1
　    no change     /lfs/mercury1/0/mariodp/miniconda/lib/python3.11/site-packages/xontrib/conda.xsh
      no change     /lfs/mercury1/0/mariodp/miniconda/etc/profile.d/conda.csh
      no change     /lfs/mercury1/0/mariodp/.bashrc
      No action taken.」'''
################################################################################################

conda install conda-build
##############################################################################################
  '''You'll see something like
 「Collecting package metadata (current_repodata.json): done
  Solving environment: done

  ## Package Plan ##

    environment location: /lfs/mercury1/0/mariodp/miniconda

    added / updated specs:
      - conda-build


  The following packages will be downloaded:

      package                    |            build
      ---------------------------|-----------------
      beautifulsoup4-4.12.2      |  py311h06a4308_0         268 KB
      chardet-4.0.0              |py311h06a4308_1003         218 KB
      click-8.1.7                |  py311h06a4308_0         221 KB
      conda-build-3.27.0         |  py311h06a4308_0         830 KB
      conda-index-0.3.0          |  py311h06a4308_0         225 KB
      filelock-3.9.0             |  py311h06a4308_0          21 KB
      jinja2-3.1.2               |  py311h06a4308_0         295 KB
      liblief-0.12.3             |       h6a678d5_0         1.9 MB
      markupsafe-2.1.1           |  py311h5eee18b_0          25 KB
      more-itertools-8.12.0      |     pyhd3eb1b0_0          49 KB
      patch-2.7.6                |    h7b6447c_1001         119 KB
      patchelf-0.17.2            |       h6a678d5_0          98 KB
      pkginfo-1.9.6              |  py311h06a4308_0          65 KB
      psutil-5.9.0               |  py311h5eee18b_0         463 KB
      py-lief-0.12.3             |  py311h6a678d5_0         1.3 MB
      python-libarchive-c-2.9    |     pyhd3eb1b0_1          47 KB
      pytz-2023.3.post1          |  py311h06a4308_0         216 KB
      pyyaml-6.0.1               |  py311h5eee18b_0         210 KB
      six-1.16.0                 |     pyhd3eb1b0_1          18 KB
      soupsieve-2.5              |  py311h06a4308_0          92 KB
      yaml-0.2.5                 |       h7b6447c_0          75 KB
      ------------------------------------------------------------
                                             Total:         6.7 MB

  The following NEW packages will be INSTALLED:

    beautifulsoup4     pkgs/main/linux-64::beautifulsoup4-4.12.2-py311h06a4308_0
    chardet            pkgs/main/linux-64::chardet-4.0.0-py311h06a4308_1003
    click              pkgs/main/linux-64::click-8.1.7-py311h06a4308_0
    conda-build        pkgs/main/linux-64::conda-build-3.27.0-py311h06a4308_0
    conda-index        pkgs/main/linux-64::conda-index-0.3.0-py311h06a4308_0
    filelock           pkgs/main/linux-64::filelock-3.9.0-py311h06a4308_0
    jinja2             pkgs/main/linux-64::jinja2-3.1.2-py311h06a4308_0
    liblief            pkgs/main/linux-64::liblief-0.12.3-h6a678d5_0
    markupsafe         pkgs/main/linux-64::markupsafe-2.1.1-py311h5eee18b_0
    more-itertools     pkgs/main/noarch::more-itertools-8.12.0-pyhd3eb1b0_0
    patch              pkgs/main/linux-64::patch-2.7.6-h7b6447c_1001
    patchelf           pkgs/main/linux-64::patchelf-0.17.2-h6a678d5_0
    pkginfo            pkgs/main/linux-64::pkginfo-1.9.6-py311h06a4308_0
    psutil             pkgs/main/linux-64::psutil-5.9.0-py311h5eee18b_0
    py-lief            pkgs/main/linux-64::py-lief-0.12.3-py311h6a678d5_0
    python-libarchive~ pkgs/main/noarch::python-libarchive-c-2.9-pyhd3eb1b0_1
    pytz               pkgs/main/linux-64::pytz-2023.3.post1-py311h06a4308_0
    pyyaml             pkgs/main/linux-64::pyyaml-6.0.1-py311h5eee18b_0
    six                pkgs/main/noarch::six-1.16.0-pyhd3eb1b0_1
    soupsieve          pkgs/main/linux-64::soupsieve-2.5-py311h06a4308_0
    yaml               pkgs/main/linux-64::yaml-0.2.5-h7b6447c_0


  Proceed ([y]/n)? y


  Downloading and Extracting Packages:

  Preparing transaction: done
  Verifying transaction: done
  Executing transaction: done」'''
####################################################################################################


conda update -n base -c defaults conda
#################################################################################################
    '''You should see something like
    「Collecting package metadata (current_repodata.json): done
      Solving environment: done

      ## Package Plan ##

      environment location: /lfs/mercury1/0/mariodp/miniconda       

      added / updated specs:
        - conda


    The following packages will be downloaded:

        package                    |            build
        ---------------------------|-----------------
        brotli-python-1.0.9        |  py311h6a678d5_7         318 KB
        libcurl-8.4.0              |       h251f7ec_0         411 KB
        libnghttp2-1.57.0          |       h2d74bed_0         674 KB
        urllib3-1.26.18            |  py311h06a4308_0         251 KB
        ------------------------------------------------------------
                                               Total:         1.6 MB

    The following NEW packages will be INSTALLED:

      brotli-python      pkgs/main/linux-64::brotli-python-1.0.9-py311h6a678d5_7

    The following packages will be REMOVED:

      brotlipy-0.7.0-py311h5eee18b_1002

    The following packages will be UPDATED:

      libcurl                                  8.2.1-h251f7ec_0 --> 8.4.0-h251f7ec_0
      libnghttp2                              1.52.0-h2d74bed_1 --> 1.57.0-h2d74bed_0        
      urllib3                           1.26.16-py311h06a4308_0 --> 1.26.18-py311h06a4308_0  


    Proceed ([y]/n)? y


    Downloading and Extracting Packages:

    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done」　'''
##################################################################################################

conda update conda
################################################################################################
    '''You should see something like:
    「Collecting package metadata (current_repodata.json): done
     Solving environment: done

     # All requested packages already installed.

############################################################################################

# - Create conda env
conda create -n my_env python=3.10
############################################################################################
    '''You should see something like:
      「Collecting package metadata (current_repodata.json): done
        Solving environment: done

        ## Package Plan ##

          environment location: /lfs/mercury1/0/mariodp/miniconda/envs/my_env

        added / updated specs:
          - python=3.10


      The following packages will be downloaded:

          package                    |            build
          ---------------------------|-----------------
          pip-23.3                   |  py310h06a4308_0         2.7 MB
          python-3.10.13             |       h955ad1f_0        26.8 MB
          setuptools-68.0.0          |  py310h06a4308_0         936 KB
          wheel-0.41.2               |  py310h06a4308_0         109 KB
          ------------------------------------------------------------
                                                 Total:        30.5 MB

      The following NEW packages will be INSTALLED:

        _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
        _openmp_mutex      pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu
        bzip2              pkgs/main/linux-64::bzip2-1.0.8-h7b6447c_0
        ca-certificates    pkgs/main/linux-64::ca-certificates-2023.08.22-h06a4308_0
        ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.38-h1181459_1
        libffi             pkgs/main/linux-64::libffi-3.4.4-h6a678d5_0
        libgcc-ng          pkgs/main/linux-64::libgcc-ng-11.2.0-h1234567_1
        libgomp            pkgs/main/linux-64::libgomp-11.2.0-h1234567_1
        libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-11.2.0-h1234567_1
        libuuid            pkgs/main/linux-64::libuuid-1.41.5-h5eee18b_0
        ncurses            pkgs/main/linux-64::ncurses-6.4-h6a678d5_0
        openssl            pkgs/main/linux-64::openssl-3.0.11-h7f8727e_2
        pip                pkgs/main/linux-64::pip-23.3-py310h06a4308_0
        python             pkgs/main/linux-64::python-3.10.13-h955ad1f_0
        readline           pkgs/main/linux-64::readline-8.2-h5eee18b_0
        setuptools         pkgs/main/linux-64::setuptools-68.0.0-py310h06a4308_0
        sqlite             pkgs/main/linux-64::sqlite-3.41.2-h5eee18b_0
        tk                 pkgs/main/linux-64::tk-8.6.12-h1ccaba5_0
        tzdata             pkgs/main/noarch::tzdata-2023c-h04d1e81_0
        wheel              pkgs/main/linux-64::wheel-0.41.2-py310h06a4308_0
        xz                 pkgs/main/linux-64::xz-5.4.2-h5eee18b_0
        zlib               pkgs/main/linux-64::zlib-1.2.13-h5eee18b_0


      Proceed ([y]/n)? y


      Downloading and Extracting Packages:

      Preparing transaction: done
      Verifying transaction: done
      Executing transaction: done
      #
      # To activate this environment, use
      #
      #     $ conda activate my_env
      #
      # To deactivate an active environment, use
      #
      #     $ conda deactivate」'''
################################################################################

conda activate my_env
## conda remove --name my_env --all

# - Make sure pip is up to date
which python
#####################################################################################
    　'''You should see something like:
      「/lfs/mercury1/0/mariodp/miniconda/envs/my_env/bin/python]
#####################################################################################

pip install --upgrade pip
###################################################################################
      '''You should see something like:
      「Requirement already satisfied: pip in ./miniconda/envs/my_env/lib/python3.10/site-packages (23.3)
       Collecting pip
        Downloading pip-23.3.1-py3-none-any.whl.metadata (3.5 kB)
      Downloading pip-23.3.1-py3-none-any.whl (2.1 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 24.8 MB/s eta 0:00:00
      Installing collected packages: pip
        Attempting uninstall: pip
          Found existing installation: pip 23.3
          Uninstalling pip-23.3:
            Successfully uninstalled pip-23.3
      Successfully installed pip-23.3.1」'''
###################################################################################

pip3 install --upgrade pip
####################################################################################
    '''「Requirement already satisfied: pip in ./miniconda/envs/my_env/lib/python3.10/site-packages (23.3.1)」'''
####################################################################################

which pip
####################################################################################
    '''「/lfs/mercury1/0/mariodp/miniconda/envs/my_env/bin/pip」'''
#######################################################################################

which pip3
#######################################################################################
    '''「/lfs/mercury1/0/mariodp/miniconda/envs/my_env/bin/pip3」'''
#######################################################################################

# - [NOT This one, full anacoda is huge] installing full anaconda
#wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh
#wget https://repo.continuum.io/conda/Anaconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
#nohup bash ~/anaconda.sh -b -p $HOME/anaconda > anaconda_install.out &
#ls -lah $HOME | grep anaconda
#source ~/anaconda/bin/activate
