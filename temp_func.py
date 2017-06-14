#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from tqdm import  tqdm

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
    for i in tqdm(range(len(f_list))):
        case = f_list[i]
        split_list = case.split('/')
        phase = split_list[-3]
        assert phase == 'train' or phase == 'test'
        label = int(split_list[-2])
        x_df = pd.read_csv(case)
        x = x_df[0][0]
        x = x.split(',')
        x = [int(item) for item in x]
        assert len(x) == 1024
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