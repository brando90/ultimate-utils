import time

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('employee.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

class Employee:
    """A sample Employee class"""

    def __init__(self, first, last):
        self.first = first
        self.last = last

        logger.info('Created Employee: {} - {}'.format(self.fullname, self.email))

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

emp_1 = Employee('John', 'Smith')
emp_2 = Employee('Corey', 'Schafer')
emp_3 = Employee('Jane', 'Doe')

######## END OF EMPLOYEE LOGGING EXAMPLE

def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    meta_str=''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    if verbose:
        print(f"--- {seconds} {'seconds '+meta_str} ---")
        print(f"--- {minutes} {'minutes '+meta_str} ---")
        print(f"--- {hours} {'hours '+meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours

def params_in_comp_graph():
    import torch
    import torch.nn as nn
    from torchviz import make_dot
    fc0 = nn.Linear(in_features=3,out_features=1)
    params = [('fc0', fc0)]
    mdl = nn.Sequential(OrderedDict(params))

    x = torch.randn(1,3)
    #x.requires_grad = True  # uncomment to put in computation graph
    y = torch.randn(1)

    l = ( mdl(x) - y )**2

    #make_dot(l, params=dict(mdl.named_parameters()))
    params = dict(mdl.named_parameters())
    #params = {**params, 'x':x}
    make_dot(l,params=params).render('data/debug/test_img_l',format='png')

def check_if_tensor_is_detached():
    a = torch.tensor([2.0], requires_grad=True)
    b = a.detach()
    b.requires_grad = True
    print(a == b)
    print(a is b)
    print(a)
    print(b)

    la = (5.0 - a)**2
    la.backward()
    print(f'a.grad = {a.grad}')

    lb = (6.0 - b)**2
    lb.backward()
    print(f'b.grad = {b.grad}')

def deep_copy_issue():
    params = OrderedDict( [('fc1',nn.Linear(in_features=3,out_features=1))] )
    mdl0 = nn.Sequential(params)
    mdl1 = copy.deepcopy(mdl0)
    print(id(mdl0))
    print(mdl0)
    print(id(mdl1))
    print(mdl1)
    # my update
    mdl1.fc1.weight = nn.Parameter( mdl1.fc1.weight + 1 )
    mdl2 = copy.deepcopy(mdl1)
    print(id(mdl2))
    print(mdl2)

def download_mini_imagenet():
    # download mini-imagenet automatically
    import torch
    import torch.nn as nn
    import torchvision.datasets.utils as utils
    from torchvision.datasets.utils import download_and_extract_archive
    from torchvision.datasets.utils import download_file_from_google_drive

    ## download mini-imagenet
    #url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    root = '~/tmp/' # dir to place downloaded file in
    download_file_from_google_drive(file_id, root, filename)

def extract():
    from torchvision.datasets.utils import extract_archive
    from_path = os.path.expanduser('~/Downloads/miniImagenet.tgz')
    extract_archive(from_path)

def download_and_extract_miniImagenet(root):
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    ## download miniImagenet
    #url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    download_file_from_google_drive(file_id, root, filename)
    fpath = os.path.join(root, filename) # this is what download_file_from_google_drive does
    ## extract downloaded dataset
    from_path = os.path.expanduser(fpath)
    extract_archive(from_path)
    ## remove the zip file
    os.remove(from_path)
    
def torch_concat():
    import torch
    
    g1 = torch.randn(3,3)
    g2 = torch.randn(3,3)


if __name__ == "__main__":
    start = time.time()
    print('pytorch playground!')
    # params_in_comp_graph()
    #check_if_tensor_is_detached()
    #deep_copy_issue()
    #download_mini_imagenet()
    #extract()
    #download_and_extract_miniImagenet(root='~/tmp')
    #download_and_extract_miniImagenet(root='~/automl-meta-learning/data')
    torch_concat()
    print('--> DONE')
    time_passed_msg, _, _, _ = report_times(start)
    print(f'--> {time_passed_msg}')