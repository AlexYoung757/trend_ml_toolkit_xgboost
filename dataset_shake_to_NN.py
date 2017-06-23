#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from tqdm import  tqdm

from tools import dirlist,to_NN

def get_config():
    config = dict()
    config['data_path'] = '/home/lili/opcode-2017-05-hash/'
    config['NN_train'] = './Data/opcode-2017-05-hash/NN_train.txt'
    config['NNAI_train'] = './Data/opcode-2017-05-hash/NNAI_train.txt'
    config['NN_test'] = './Data/opcode-2017-05-hash/NN_test.txt'
    config['NNAI_test'] = './Data/opcode-2017-05-hash/NNAI_test.txt'
    return config

def get_sample_label(f_list):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    path_train = []
    path_test = []

    for i in tqdm(range(len(f_list))):
        case = f_list[i]
        split_list = case.split('/')
        phase = split_list[-3]
        assert phase == 'train' or phase == 'test'
        label = int(split_list[-2])
        assert label == 1 or label == 0
        with open(case,mode='r') as f:
            x = f.readlines() # 只有一行
            x = x[1]
            x = x.split(',')
            x = [int(item) for item in x]
            assert len(x) == 1024

        if(phase == 'train'):
            X_train.append(x)
            Y_train.append(label)
            path_train.append(case)
        else:
            X_test.append(x)
            Y_test.append(label)
            path_test.append(case)

    return np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test),np.array(path_train),np.array(path_test)

if __name__ == '__main__':
    config = get_config()

    f_list = dirlist(config['data_path'],[])
    X_train,Y_train,X_test,Y_test,path_train,path_test = get_sample_label(f_list)
    to_NN(data=X_train,label=Y_train,path=path_train,NN_name=config['NN_train'],NN_label_name=config['NNAI_train'])
    to_NN(data=X_test,label=Y_test,path=path_test,NN_name=config['NN_test'],NN_label_name=config['NNAI_test'])
    # dump_svmlight_file(X=X_train,y=Y_train,f=config['re_path']+'train.libsvm',zero_based=True)
    # dump_svmlight_file(X=X_test,y=Y_test,f=config['re_path']+'test.libsvm',zero_based=True)