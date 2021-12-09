## PURPOSE: print the path plus the file name to be opened by docker vim.
## The intention is that this is the argument to the vim command executed by docker
## (as in docker ... vim OUTPUT_OF_pvim.py ). Thus, the output of pvim.py is a string
## with path+filename so that when docker (and the vim command it executes) gets the
## string, it knows the path and file to open with vim. The path needs be absolute with respect
## to the paths in the docker container (not the original host). Thus, it needs to
## strip of the prefix and find 'home_simulation_research' so that it knows where
## the path to the file in the container starts.

import sys
from subprocess import call

def hello_world():
    print('hello world')

def _make_prefix(pwd_array):
    '''
    Given pwd_array concatenates all the elements of the array and puts a / between them.

    e.g.
    input = [ "home_simulation_research", "hbf_tensorflow_code", "docker_files", "vim_img"]
    output = "/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/"
    '''
    #print(pwd_array)
    string = ''
    for path_part in pwd_array:
        #print(path_part)
        string = string + '/' + path_part
    string = string + '/'
    return string

def get_prefix_for_docker_env(pwd):
    '''
    Returns the prefix string relative to my  docker filesystem.

    e.g.
    input = '/Users/brandomiranda/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img'
    output = /home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/
    '''
    pwd_split = pwd.split('/')
    for i in range( len(pwd_split) ):
        path_part = pwd_split[i]
        #print(path_part)
        if path_part == 'home_simulation_research':
            pwd_split_str = _make_prefix(pwd_split[i:])
            return pwd_split_str
    raise ValueError('SHOULD NEVER BE HERE. Probably gave a path with no home_simulation_research directory.')

def print_path_plus_filename(pwd,filename):
    pwd_split_str = get_prefix_for_docker_env(pwd)
    return pwd_split_str+filename

####
####

if __name__ == '__main__':
    pwd, filename = sys.argv[1], sys.argv[2]
    if len(sys.argv) < 3:
        raise ValueError('Need to provide a file to vim.')
    path_plus_filename = print_path_plus_filename(pwd,filename)
    print(path_plus_filename)
