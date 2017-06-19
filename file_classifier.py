#coding=utf-8
import pandas as pd
import shutil
import numpy as np
import os
import glob
from pprint import pprint
from multiprocessing import Pool
import multiprocessing
from functools import partial

def start_process():
    print 'Starting',multiprocessing.current_process().name

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def search_mv(i,refer,files):
    print("%d / %d\n" % (i, len(refer)))
    item = refer[i]
    file_name = item[0] + '.opcode'
    flag = bool(item[1])
    cases = files[files[:, 1] == file_name]
    if (len(cases) != 0):
        case_file = cases[0]
        if (phase == 'train'):
            re_path = result_path0 if flag == 0  else result_path1
        else:
            re_path = result_path2 if flag == 0  else result_path3
        shutil.copy(case_file[0], re_path)

if __name__ == '__main__':

    phase = 'test'

    handle_path = '/macml-data/features/opcode'

    result_path0 = '/home/lili/opcode-2017-05/train/0'
    result_path1 = '/home/lili/opcode-2017-05/train/1'
    result_path2 = '/home/lili/opcode-2017-05/test/0'
    result_path3 = '/home/lili/opcode-2017-05/test/1'

    result_paths = [result_path0,result_path1,result_path2,result_path3]

    for r in result_paths:
        if(os.path.exists(r) == False):
            os.makedirs(r)
            pass

    train_csv = '/home/lili/datasets/2017-05_train.csv'
    test_csv = '/home/lili/datasets/2017-05_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_np = train_df.as_matrix()
    test_np = test_df.as_matrix()

    files = dirlist(handle_path,[])
    files = np.array(files)
    files = files[:,np.newaxis]
    files = np.concatenate((files,files),axis=1)
    for i in range(len(files)):
        files[i,1] = files[i,1].split('/')[-1]
    if(phase == 'train'):
        refer = train_np
    else:
        refer = test_np

    partial_search_mv = partial(search_mv,refer=refer,files=files)
    pool = Pool(processes=None,initializer=start_process)
    pool.map(partial_search_mv,range(len(refer)))
    pool.close()
    pool.join()

    """
    for i in range(len(refer)):
        print("%d / %d\n"%(i,len(refer)))
        item = refer[i]
        file_name = item[0]+'.opcode'
        flag = bool(item[1])
        cases = files[files[:,1] == file_name]
        if(len(cases)!=0):
            case_file = cases[0]
            re_path = ''
            if(phase=='train'):
                re_path = result_path0 if flag == 0  else result_path1
            else:
                re_path = result_path2 if flag == 0  else result_path3
            shutil.copy(case_file[0],re_path)
    print('refer  :  %d\n'%(len(refer)))
    print('origin :  %d\n'%(len(files)))
    """
