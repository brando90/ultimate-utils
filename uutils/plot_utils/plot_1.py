'''
# something wrong with the logging/the way to write as a 
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
    plt.axis( [0, 10000, 0, 2.5] )    # order: min(x), max(x), min(y), max(y) ...  # the number of max(x)
    print (value_names)
    for value_list in value_lists:
        print ("The length of value_list is {}".format( len(value_list) ) )

    if 'inner_train_loss' in value_names:
        inner_iter_list = range( 0, len(value_lists[0]) )
        inner_iter_list = [ i*2 for i in inner_iter_list]
        inner_loss_id = value_names.index('inner_train_loss')
        print ("The length of inner_loss is {}; inner_loss_iter_length is {}".format( len(value_lists[inner_loss_id]), len(inner_iter_list) ))
        value_list = value_lists[inner_loss_id]
        ax.plot(inner_iter_list, value_list, 'g+', label = 'inner_train_loss')

    if 'inner_train_acc' in value_names or 'outer_train_acc' in value_names:
        ax_2 = ax.twinx()
        ax_2.set_ylabel('acc')

    if 'inner_train_acc' in value_names:
        inner_acc_id = value_names.index('inner_train_acc')
        value_list = value_lists[inner_acc_id]
        
        print ("The length of train_acc is {}; train_acc_iter_length is {}".format( len(value_lists[inner_acc_id]), len(inner_iter_list) ))
        ax_2.plot(inner_iter_list, value_list, 'y+', linewidth=10, alpha=0.7, label = 'inner_train_acc')
    else:
        print("There is no train_acc detected ... ")

    if 'outer_train_acc' in value_names:  # if we have test loss then we have test_acc
        outer_acc_id = value_names.index('outer_train_acc')
        value_list = value_lists[outer_acc_id]
        outer_iter_list = range(1, len(value_list)+1 )
        outer_iter_list = [ i*16 for i in outer_iter_list]  # modification needed here ... 
        print ("The length of outer_acc is {}; outer_acc_iter_length is {}".format( len(value_lists[outer_acc_id]), len(outer_iter_list) ))
        ax_2.plot(outer_iter_list, value_list, 'r*', linewidth=10, alpha=0.7, label = 'outer_train_acc' )
    else:
        print("There is no test_acc detected ... ")

    if 'outer_train_loss' in value_names:
        outer_loss_id = value_names.index('outer_train_loss')
        value_list = value_lists[outer_loss_id]
        print ("The length of outer_loss is {}; outer_loss_iter_length is {}".format( len(value_lists[outer_loss_id]), len(outer_iter_list) ))
        ax.plot( outer_iter_list, value_list, 'b*', label = 'outer_train_loss' )
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
in this case, all the stats are about the 
'''
def filter_out(value_list, step = 16):
    length = len(value_list)
    new_value_list = [ value_list[idx] for idx in range(0,length, step) ]
    return new_value_list


def main(save_path):
    fig_suffix = args.file_name # use the same name as logging file ...
    txt_contents = open(txt_file, 'r').read()   # wait for further notice ...
    value_names = []
    value_lists = []

    if args.train_loss == True:
        value_names.append('inner_train_loss')
        re_exp = r'inner_loss: (.*?),'
        train_loss = get_value_list(re_exp, txt_contents)
        print (len(train_loss))
        train_loss = [ float(i) for i in train_loss ]
        train_loss = filter_out(train_loss)
        value_lists.append(train_loss)

    if args.test_loss == True:
        re_exp = r'outer_loss: (.*?),'  
        test_loss = get_value_list(re_exp, txt_contents)
        value_names.append('outer_train_loss')
        test_loss = [ float(i) for i in test_loss ]
        test_loss = filter_out(test_loss)
        value_lists.append(test_loss)

    if args.train_acc == True:
        re_exp = r'inner_train_acc: (.*?),'  
        train_acc = get_value_list(re_exp, txt_contents)
        # print (train_acc) # there are normal training values ...
        value_names.append('inner_train_acc')
        train_acc = [ float(i) for i in train_acc ]
        train_acc = filter_out(train_acc)
        value_lists.append(train_acc)

    if args.test_acc == True:
        # here the re match has some problems ... 
        re_exp = r'outer_acc: (.*?),'  
        test_acc = get_value_list(re_exp, txt_contents)
        # print (test_acc)
        value_names.append('outer_train_acc')
        test_acc = [ float(i) for i in test_acc ]
        test_acc = filter_out(test_acc)
        value_lists.append(test_acc)

    plot(value_lists, value_names , save_path)   # draw those training&test stats together ...


if __name__ == "__main__":
    main( save_path )




