#coding=utf-8
import pandas as pd
import shutil
import numpy as np
import os
import glob
from pprint import pprint

if __name__ == '__main__':

    phase = 'train'

    result_path0 = '/home/lili/opcode-2017-05/train/0'
    result_path1 = '/home/lili/opcode-2017-05/train/1'
    result_path2 = '/home/lili/opcode-2017-05/test/0'
    result_path3 = '/home/lili/opcode-2017-05/test/1'

    result_paths = [result_path0,result_path1,result_path2,result_path3]

    for r in result_paths:
        if(os.path.exists(r) == False):
            os.makedirs(r)

    train_csv = '/home/lili/datasets/2017-05_train.csv'
    test_csv = '/home/lili/datasets/2017-05_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_np = train_df.as_matrix()
    test_np = test_df.as_matrix()

    handle_path = '/macml-data/features/opcode'
    now_path = os.path.abspath(handle_path)
    subfolders = os.listdir(now_path)
    search_paths = []
    for p in subfolders:
        temp = os.path.join(now_path, p)
        if(os.path.isdir(temp)):
            search_paths.append(temp)
    files =  []
    for p in search_paths:
        files_per_dir = glob.glob(p+'/*')
        files.extend(files_per_dir)
    files = np.array(files)
    files = files[:,np.newaxis]
    files = np.concatenate((files,files),axis=1)
    for i in range(len(files)):
        files[i,1] = files[i,1].split('/')[-1]

    if(phase == 'train'):
        refer = train_np
    else:
        refer = test_np

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
            print('%s  ====>  %s\n'%(case_file[0],re_path))
            shutil.copy(case_file[0],re_path)