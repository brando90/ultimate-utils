"""
create coqgym data_lib set.

Would be nice to make all the code in either python or bash but not mix it like it's done here.
Also using pathlib as much as possible.
"""
import os
import platform
import sys
from hashlib import md5
import pdb

from pathlib import Path


def execute(cmd):
    print(f'{cmd}\n')
    assert os.system(cmd) == 0

def check_md5(filename, gt_hashcode):
    print('checking %s..' % filename)
    if not os.path.exists(filename):
        print(filename, 'not exists')
        print('aborting..')
        sys.exit(-1)
    hashcode = md5(open(filename, 'rb').read()).hexdigest()
    if hashcode != gt_hashcode:
        print(filename, 'has the wrong MD5 hashcode')
        print('expect %s but found %s' % (gt_hashcode, hashcode))
        print('aborting..')
        sys.exit(-1)
    print(f'pass md5 check {filename}, {gt_hashcode}\n')
    return


def unzip(filename, extract_dir):
    """
    https://stackoverflow.com/questions/3451111/unzipping-files-in-python
    """
    filename = str(filename)
    print(f'unzipping {filename}...')
    if os.path.exists(filename[:-7]):
        # remove = input(filename[:-7] + ' already exists. Do you want to remove it? (y/N)').lower()
        remove = 'y'
        if remove == 'y':
            execute('rm -r ' + filename[:-7])
        else:
            print('aborting..')
            sys.exit(-1)

    import shutil
    shutil.unpack_archive(filename, extract_dir)

    # execute(f'tar -xvzf {filename}')
    print(f'done unzipping {filename}\n')


def do_mdb_load(sexp_cache_gz, extract_dir, original_code=False):
    """

    command:
        mdb_load [-V] [-f file] [-n] [-s subdb] [-N] [-T]  envpath
    description:
        The mdb_load utility reads from the standard input and loads it into the LMDB environment envpath.
    Kaiyu's example:
        execute('mdb_load -f sexp_cache.lmdb sexp_cache')
    """
    if original_code:
        unzip('sexp_cache.tar.gz')
        os.mkdir('sexp_cache')
        execute('mdb_load -f sexp_cache.lmdb sexp_cache')
        os.remove('sexp_cache.lmdb')
    else:
        unzip(sexp_cache_gz, extract_dir)

        sexp_cache_dir = Path('~/data/coqgym/sexp_cache').expanduser()
        sexp_cache_dir.mkdir(parents=True, exist_ok=True)

        sexp_cache_lmdb = Path('~/data/coqgym/sexp_cache.lmdb').expanduser()
        sexp_cache = Path('~/data/coqgym/sexp_cache').expanduser()
        execute(f'mdb_load -f {sexp_cache_lmdb} {sexp_cache}')

        os.remove(sexp_cache_lmdb)
    print(f'done mdb_load: {sexp_cache_gz, extract_dir}\n')

def main():
    data_root_path = Path('~/data/coqgym/').expanduser()

    projs_split = data_root_path / Path('projs_split.json')
    data = data_root_path / Path('data_lib.tar.gz')
    sexp_cache_gz = data_root_path / Path('sexp_cache.tar.gz')

    # check data_lib integrity
    check_md5(projs_split, '39eac2315532040f370ca4996862ef75')
    # check_md5(data_lib, '922937155a199605eb8067ccfbbdb81a')
    check_md5(sexp_cache_gz, '2e8ff40a7dd0b6d0efc74480dd3dfc8d')

    # unzip
    unzip(data, data_root_path)

    # create sexp
    print('doing the sexp...')
    do_mdb_load(sexp_cache_gz, data_root_path)
    print('done doing the sexps...')

    # print('setting the absolute paths...')
    # cwd = os.getcwd()
    # if platform.system() == 'Darwin':
    #     cmd = 'find ./data_lib -type f -exec sed -i \'\' \'s/TAPAS_ROOT_ABSOLUTE_PATH/%s/g\' {} +' % cwd.replace(os.path.sep, '\/')
    # else:
    #     cmd = 'find ./data_lib -type f -exec sed -i \'s/TAPAS_ROOT_ABSOLUTE_PATH/%s/g\' {} +' % cwd.replace(os.path.sep, '\/')
    # execute(cmd)


if __name__ == '__main__':
    main()
    print('The CoqGym dataset is ready!')
