'''
Usage:
be sure create a 'logs' folder in the 'utils', then save all the training logs in the terminal as a txt.file in 'olgs' folder
using command 'python plots.py -file_name your_txt_log_file_name.txt' to draw figures based on your training logs.
'''

import os
import re
import time
import numpy as np 
import matplotlib.pyplot as plt 

import argparse

# original settings ...
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type = str, default = 0, help = 'The name of txt log file. ' )
parser.add_argument('--train_loss', type = bool, default = True, help = 'Whether draw the training loss plot')
parser.add_argument('--test_loss', type = bool, default = False, help = 'Whether draw the test_loss plot')
parser.add_argument('--train_error', type = bool, default = False, help = 'Whether draw the train_acc plot')
parser.add_argument('--test_error', type = bool, default = False, help = 'Whether draw the test_acc plot')
args = parser.parse_args()

txt_file = os.path.join( os.getcwd(), 'logs', '{}'.format( args.file_name) )
save_path = os.path.join( os.getcwd(), 'plot_figures')

# this func is to plot the curves of the training loss, test loss, train_acc and test_acc with a specific name ...
def plot(value_list, value_name, save_path):
    iter_list = range( 0, len(value_list) )
    plt.figure()
    plt.title('training_curve')
    plt.xlabel('iteration')
    plt.ylabel(value_name + '(log)')
    plt.axis( [-1, len(value_list)+1, min(value_list)-0.5, 3.5] )
    # plt.axis( [ -1, len(value_list)+1, min(value_list), max(value_list) ] )  # take care of the settings here ...
    plt.plot(iter_list, value_list, 'r*')
    # plt.yscale('log')
    plt.show()
    plt.savefig(save_path + '/' + value_name + '_local.jpg')
 

def get_value_list(txt_file, re_exp):
    value_list = re.findall(txt_file, re_exp)
    return value_list

'''
episode/outer_i = 0
[e=outer_i=0], meta_loss: 2.2095298767089844, train error: -1, test loss: -1, test error: -1

episode/outer_i = 1
[e=outer_i=1], meta_loss: 2.2083024978637695, train error: -1, test loss: -1, test error: -1

episode/outer_i = 2
[e=outer_i=2], meta_loss: 2.2070059776306152, train error: -1, test loss: -1, test error: -1
'''

# test for the first ten lines from training output logs ... 
def main(save_path):
    fig_suffix = args.file_name
    txt_contents = open(txt_file, 'r').read()   # wait for further notice ...

    if args.train_loss == True:
        re_exp = r'meta_loss: (.*?), train error:'  # for now ...
        train_loss = get_value_list(re_exp, txt_contents)
        '''
        the extracted meta_train_losses are in the right order ...
        print ("the train_losses are : \n")
        print (train_loss)
        '''
        train_loss = [ float(i) for i in train_loss ]
        # map( list(float, list_A) )  # this is the map ...
        plot(train_loss, 'train_loss_' + str(fig_suffix) , save_path)

    if args.test_loss == True:
        re_exp = None  # haven't define it in the program ... 
        test_loss = get_value_list(re_exp, txt_contents)
        plot(test_loss, 'test_loss_' + str(fig_suffix) , save_path)

    if args.train_error == True:
        re_exp = r'train error: (.*?), test loss:'  
        train_acc = get_value_list(re_exp, txt_contents)
        plot(train_acc, 'train_error_' + str(fig_suffix) , save_path)

    if args.test_error == True:
        re_exp = r'test error: (.*?)\n' 
        test_acc = get_value_list(re_exp, txt_contents)
        plot(test_acc, 'test_error_' + str(fig_suffix) , save_path)


if __name__ == "__main__":
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    main( save_path )




