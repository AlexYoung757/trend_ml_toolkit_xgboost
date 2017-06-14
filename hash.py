#coding=utf-8

import os
import hashlib
import sha3
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from functools import partial

def start_process():
    print 'Starting',multiprocessing.current_process().name

def hexstr2bitstr(string):
    hex_list = [format(int(c,16),'b') for c in string ]
    re_list = []
    for l in hex_list:
	len_l = len(l)
	for i in range(4-len_l):
	    l = '0'+l
	assert len(l) == 4
	re_list.append(l)
    temp = ''.join(re_list)
    temp = [c for c in temp]
    return temp

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def get_config():
    config = dict()
    config['handle_path'] = '/home/lili/opcode-2017-05'
    config['re_path'] = '/home/lili/opcode-2017-05-hash2'
    config['length'] = 1024
    return config

def handle_f(i,files,config):
    print('%d / %d'%(i,len(files)))
    f_full_name = files[i]
    f_name = f_full_name.split('/')[-1]
    flag = f_full_name.split('/')[-2]
    assert flag == '0' or flag == '1'
    phase = f_full_name.split('/')[-3]
    assert phase == 'train' or phase == 'test'
    with open(f_full_name) as f:
        content = f.read()
    fea_str = hashlib.shake128(config['length'], content).hexdigest()
    fea_str_bit = hexstr2bitstr(fea_str)
    fea_str_bit = np.array(fea_str_bit)
    fea_str_bit = fea_str_bit.reshape((1, -1))
    assert fea_str_bit.shape[1] == config['length']
    base_path = os.path.join(config['re_path'], phase + '/' + flag)
    if (os.path.exists(base_path) == False):
        os.makedirs(base_path)
    fea_names = ['f' + str(i) for i in range(config['length'])]
    r = pd.DataFrame(data=fea_str_bit, columns=fea_names)
    r.to_csv(base_path + '/' + f_name+'.csv')
    #print(r.columns)
    assert len(r.columns == config['length'])
    #print(r.columns)
    #print(len(r[0][0]))

if __name__ == '__main__':

    config = get_config()
    files = dirlist(config['handle_path'],[])
    N = len(files)
    partial_handle_f = partial(handle_f,files=files,config=config)
    pool = Pool(processes=None,initializer=start_process)
    pool_outputs=pool.map(partial_handle_f,range(N))
    pool.close()
    pool.join()
"""
    for i in tqdm(range(N)):
        f_full_name = files[i]
        f_name = f_full_name.split('/')[-1]
        flag = f_full_name.split('/')[-2]
        assert flag == '0' or flag == '1'
        phase = f_full_name.split('/')[-3]
        assert phase == 'train' or phase == 'test'
        with open(f_full_name) as f:
            content = f.read()
        fea_str = hashlib.shake128(config['length'],content).hexdigest()
        fea_str_bit = hexstr2bitstr(fea_str)
	print(len(fea_str_bit))
        fea_str_bit = np.array(fea_str_bit)
        fea_str_bit = fea_str_bit.reshape((1,-1))
	print(fea_str_bit.shape)
        assert fea_str_bit.shape[1] == config['length']
        base_path = os.path.join(config['re_path'],phase+'/'+flag)
        if(os.path.exists(base_path)==False):
            os.makedirs(base_path)
        fea_names = [ 'f'+str(i) for i in range(config['length'])]
        r = pd.DataFrame(data=fea_str_bit,columns=fea_names)
        r.to_csv(base_path+'/'+f_name+'.csv')
        print(r)
        print(r.columns)
"""
