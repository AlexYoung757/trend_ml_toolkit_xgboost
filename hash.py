#coding=utf-8

import os
import hashlib
import sha3
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pandas as pd

def start_process():
    print 'Starting',multiprocessing.current_process().name

def hexstr2bitstr(string):
    hex_list = [format(int(c,16),'b') for c in string ]
    return ''.join(hex_list)

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
    config['handle_path'] = '/home/lili/opcode-07'
    config['re_path'] = '/home/lili'
    config['length'] = 1024

def handle_f(i,files,config):
    f_full_name = files[i]
    f_name = f_full_name.split('/')[-1]
    flag = f_full_name.split('/')[-2]
    assert flag == '0' or flag == '1'
    phase = f_full_name.split('/')[-3]
    assert flag == 'train' or flag == 'test'
    with open(f_full_name) as f:
        content = f.read()
    fea_str = hashlib.shake_128(content).hexdigest(length=config['length'] / 8)
    fea_str_bit = hexstr2bitstr(fea_str)
    fea_str_bit = np.array(fea_str_bit)
    fea_str_bit = fea_str_bit.reshape((1, -1))
    assert fea_str_bit.shape[1] == config['length']
    base_path = os.path.join(config['re_path'], phase + '/' + flag)
    if (os.path.exists(base_path) == False):
        os.makedirs(base_path)
    fea_names = ['f' + str(i) for i in range(config['length'])]
    r = pd.DataFrame(data=fea_str_bit, columns=fea_names)
    r.to_csv(base_path + '/' + f_name)

if __name__ == '__main__':

    config = get_config()
    files = dirlist(config['handle_path'],[])
    N = len(files)
    pool = Pool(processes=None,initializer=start_process)
    pool_outputs=pool.map(handle_f(files=files,config=config),range(N))
    pool.close()
    pool.join()
    """
    for i in tqdm(range(N)):
        f_full_name = files[i]
        f_name = f_full_name.split('/')[-1]
        flag = f_full_name.split('/')[-2]
        assert flag == '0' or flag == '1'
        phase = f_full_name.split('/')[-3]
        assert flag == 'train' or flag == 'test'
        with open(f_full_name) as f:
            content = f.read()
        fea_str = hashlib.shake_128(content).hexdigest(length=config['length']/8)
        fea_str_bit = hexstr2bitstr(fea_str)
        fea_str_bit = np.array(fea_str_bit)
        fea_str_bit = fea_str_bit.reshape((1,-1))
        assert fea_str_bit.shape[1] == config['length']
        base_path = os.path.join(config['re_path'],phase+'/'+flag)
        if(os.path.exists(base_path)==False):
            os.makedirs(base_path)
        fea_names = [ 'f'+str(i) for i in range(config['length'])]
        r = pd.DataFrame(data=fea_str_bit,columns=fea_names)
        r.to_csv(base_path+'/'+f_name)
    """
