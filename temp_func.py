#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file

from tools import dirlist

def get_config():
    config = dict()
    config['data_path'] = '/home/lili/opcode-2017-05-hash/'
    config['re_path'] = './Output/'
    return config

def get_sample_label(f_list):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for case in f_list:
        split_list = case.split('/')
        phase = split_list[-3]
        label = int(split_list[-2])
        with open(case,mode='r') as f:
            x = f.read()
            x = x.split(',')
            x = [int(item) for item in x]
            assert len(x) == 1024
        assert phase == 'train' or phase == 'test'
        if(phase == 'train'):
            X_train.append(x)
            Y_train.append(label)
        else:
            X_test.append(x)
            Y_test.append(label)
    return np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)

if __name__ == '__main__':
    config = get_config()

    f_list = dirlist(config['data_path'],[])
    X_train,Y_train,X_test,Y_test = get_sample_label(f_list)
    dump_svmlight_file(X=X_train,y=Y_train,f=config['re_path']+'train.libsvm',zero_based=True)
    dump_svmlight_file(X=X_test,y=Y_test,f=config['re_path']+'test.libsvm',zero_based=True)