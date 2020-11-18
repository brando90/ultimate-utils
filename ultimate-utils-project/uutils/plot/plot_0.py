'''
this is a new script to plot training loss&acc, test loss and acc ... 
this script is based on the new logger (from meta-lstm ... ) and the figure will contain more info ...

by the way, look at the logs of meta-lstm program, we would understand that ... 
what are the contents within the brackets ... ???
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
parser.add_argument('--test_loss', type = bool, default = True, help = 'Whether draw the test_loss plot')
parser.add_argument('--train_acc', type = bool, default = True, help = 'Whether draw the train_acc plot')
parser.add_argument('--test_acc', type = bool, default = True, help = 'Whether draw the test_acc plot')
args = parser.parse_args()

txt_file = os.path.join( os.getcwd(), 'logs', '{}.txt'.format( args.file_name) )
save_path = os.path.join( os.getcwd(), 'plot_figures')


def plot(value_lists, value_names, save_path):

    fig, ax = plt.subplots()
    ax.set_title('Training_Curve')
    plt.xlabel('iteration/episode_number')

    ax.set_ylabel('loss value')
    plt.axis( [0, 50000, 0, 2.5] )    # order: min(x), max(x), min(y), max(y) ...
    print (value_names)
    for value_list in value_lists:
        print ("The length of value_list is {}".format( len(value_list) ) )

    if 'train_loss' in value_names:
        train_iter_list = range( 0, len(value_lists[0]) )
        train_iter_list = [ i*50 for i in train_iter_list]
        train_loss_id = value_names.index('train_loss')
        print ("The length of train_loss is {}; train_loss_iter_length is {}".format( len(value_lists[train_loss_id]), len(train_iter_list) ))
        value_list = value_lists[train_loss_id]
        ax.plot(train_iter_list, value_list, 'g+', label = 'train_loss')

    if 'train_acc' in value_names or 'test_acc' in value_names:
        ax_2 = ax.twinx()
        ax_2.set_ylabel('acc')

    if 'train_acc' in value_names:
        train_acc_id = value_names.index('train_acc')
        value_list = value_lists[train_acc_id]
        
        print ("The length of train_acc is {}; train_acc_iter_length is {}".format( len(value_lists[train_acc_id]), len(train_iter_list) ))
        ax_2.plot(train_iter_list, value_list, 'y+', linewidth=10, alpha=0.7, label = 'train_acc')
    else:
        print("There is no train_acc detected ... ")

    if 'test_acc' in value_names:  # if we have test loss then we have test_acc
        test_acc_id = value_names.index('test_acc')
        value_list = value_lists[test_acc_id]
        test_iter_list = range(1, len(value_list)+1 )
        test_iter_list = [ i*1000 for i in test_iter_list]
        print ("The length of test_acc is {}; test_acc_iter_length is {}".format( len(value_lists[test_acc_id]), len(test_iter_list) ))
        ax_2.plot(test_iter_list, value_list, 'r*', linewidth=10, alpha=0.7, label = 'test_acc' )
    else:
        print("There is no test_acc detected ... ")

    if 'test_loss' in value_names:
        test_loss_id = value_names.index('test_loss')
        value_list = value_lists[test_loss_id]
        print ("The length of test_loss is {}; test_loss_iter_length is {}".format( len(value_lists[test_loss_id]), len(test_iter_list) ))
        ax.plot( test_iter_list, value_list, 'b*', label = 'test_loss' )
    else:
        print("There is no test_loss detected ... ")

    ax.legend(loc='upper left')
    ax_2.legend(loc='upper right')
    ax.yaxis.label.set_color('blue')
    ax_2.spines['left'].set_color('blue')
    ax_2.yaxis.label.set_color('red')
    ax_2.spines['right'].set_color('red')
    plt.show()
    plt.savefig(save_path + '/' + args.file_name + '.jpg')
 

def get_value_list(txt_file, re_exp):
    # here, we use re to match all contents from global context ... 
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


'''
double axis standards; makers for different lines ... 
the right x value in the chart ... 
'''
def main(save_path):
    fig_suffix = args.file_name # use the same name as logging file ...
    txt_contents = open(txt_file, 'r').read()   # wait for further notice ...
    value_names = []
    value_lists = []

    if args.train_loss == True:
        value_names.append('train_loss')
        re_exp = r'] loss: (.*?) '
        train_loss = get_value_list(re_exp, txt_contents)
        train_loss = [ float(i) for i in train_loss ]
        value_lists.append(train_loss)

    if args.test_loss == True:
        re_exp = r'Eval [(]100 episode[)] - loss: (.*?) [+]-'  
        test_loss = get_value_list(re_exp, txt_contents)
        value_names.append('test_loss')
        test_loss = [ float(i) for i in test_loss ]
        value_lists.append(test_loss)

    if args.train_acc == True:
        re_exp = r', acc: (.*?)% '  
        train_acc = get_value_list(re_exp, txt_contents)
        # print (train_acc) # there are normal training values ...
        value_names.append('train_acc')
        train_acc = [ float(i) * 0.01 for i in train_acc ]
        value_lists.append(train_acc)

    if args.test_acc == True:
        # here the re match has some problems ... 
        re_exp = r'acc: (.*?) [+]-'  
        test_acc = get_value_list(re_exp, txt_contents)
        # print (test_acc)
        value_names.append('test_acc')
        test_acc = [ float(i) * 0.01 for i in test_acc ]
        value_lists.append(test_acc)

    plot(value_lists, value_names , save_path)   # draw those training&test stats together ...


if __name__ == "__main__":
    main( save_path )
