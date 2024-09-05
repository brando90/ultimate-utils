import time
import os

from pathlib import Path


def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"
    :param float start: the number representing start (usually time.time())
    '''
    meta_str = ''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds / 60
    hours = minutes / 60
    if verbose:
        print(f"--- {seconds} {'seconds ' + meta_str} ---")
        print(f"--- {minutes} {'minutes ' + meta_str} ---")
        print(f"--- {hours} {'hours ' + meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours


def download_and_extract_miniimagenet(path):
    """ Function to download miniImagent from google drive link.
    sources:
    - https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR
    - https://github.com/markdtw/meta-learning-lstm-pytorch
    """
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    path = path.expanduser()

    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename_zip = 'miniImagenet.tgz'
    # if zip not there re-download it
    path_2_zip = path / filename_zip
    if not path_2_zip.exists():
        download_file_from_google_drive(file_id, path, filename_zip)
    else:
        print(f'Zip file to data set is already there at {path_2_zip}')
    # if actual data is not in appriopriate location extract it from zip to location
    extracted_dataset_path = path / 'miniImagenet'
    if not extracted_dataset_path.exists():
        # improve: https://stackoverflow.com/questions/3451111/unzipping-files-in-python
        os.system(
            f'tar -xvzf {path_2_zip} -C {path}/')  # extract data set in above location i.e at path / 'miniImagenet'
    else:
        print(f'Extracted data set from zip is there already! Check it at: {extracted_dataset_path}')


def download_and_extract_miniimagenet_rfs(path):
    """ Function to download miniImagent from google drive link.
    sources:
        - original: https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0
    """
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    path = path.expanduser()

    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename_zip = 'miniImageNet.tar.gz'
    # if zip not there re-download it
    path_2_zip = path / filename_zip
    if not path_2_zip.exists():
        # download_file_from_google_drive(file_id, path, filename_zip)
        assert (False)
    else:
        print(f'Zip file to data set is already there at {path_2_zip}')
    # if actual data is not in appriopriate location extract it from zip to location
    extracted_dataset_path = path / 'miniImagenet'
    if not extracted_dataset_path.exists():
        # improve: https://stackoverflow.com/questions/3451111/unzipping-files-in-python
        os.system(
            f'tar -xvzf {path_2_zip} -C {path}/')  # extract data set in above location i.e at path / 'miniImagenet'
    else:
        print(f'Extracted data set from zip is there already! Check it at: {extracted_dataset_path}')


if __name__ == "__main__":
    start = time.time()
    print('-> starting Downlooad')

    # dir to place mini-imagenet
    # path = Path('~/data/miniimagenet_meta_lstm/').expanduser()
    path = Path('~/data/miniImageNet_rfs/').expanduser()
    print(f'download path = {path}')
    # download_and_extract_miniimagenet(path)
    download_and_extract_miniimagenet_rfs(path)

    print('--> DONE')
    time_passed_msg, _, _, _ = report_times(start)
    print(f'--> {time_passed_msg}')
