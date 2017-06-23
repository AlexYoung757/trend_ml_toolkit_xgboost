import scipy.sparse
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sample', required=True)
    parser.add_argument('-l','--label', required=True)
    # parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

def get_csr_labels(xgmat_dataset_path):
    print ('start running example of build DMatrix from scipy.sparse CSR Matrix')
    labels = []
    row = []; col = []; dat = []
    i = 0
    for l in open(xgmat_dataset_path):
        arr = l.split()
        labels.append(int(arr[0]))
        for it in arr[1:]:
            k,v = it.split(':')
            row.append(i); col.append(int(k)); dat.append(float(v))
        i += 1
    csr = scipy.sparse.csr_matrix((dat, (row,col)))

    return csr,labels

def get_predictedResults_from_file(file_path):
    labels = []
    preds_score = []
    label = 0
    with open(file_path) as data_file:
        lines = data_file.readlines()
        # num = len(lines)
        # print num
        for line in lines:
            if(line.find('bad')!=-1):   label = 1
            if(line.find('good')!=-1):  label = 0
            score = float(line.split(' ')[-1])
            # print score,label
            preds_score.append(score)
            labels.append(label)
    # print labels,preds_score
    return labels,preds_score

def tune_classifier(estimator,params,X_train,Y_train,scoring='roc_auc',n_jobs=3,cv=5):
    results = []
    for k,values in params.items():
        params_single = dict(k=values)
        print '==========  ',params_single,'  =============='
        grid_search = GridSearchCV(estimator,param_grid=params_single,scoring=scoring,n_jobs=n_jobs,cv=cv,verbose=5)
        grid_search.fit(X_train,Y_train)
        df0 = pd.DataFrame(grid_search.cv_results_)
        df = pd.DataFrame(grid_search.cv_results_)[['params','mean_train_score','mean_test_score']]
        # print df0
        print df
        print 'the best_params : ',grid_search.best_params_
        print 'the best_score  : ',grid_search.best_score_
        # print grid_search.cv_results_
        results.append(grid_search.best_params_)
    return results


def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

# read path
def read_sample_path(path):
    with open(path, 'r') as label_fi:
        paths = []
        for line in label_fi.readlines():
            temp = line.split('|')[1]
            paths.append(temp)
        sample_paths = np.asarray(paths)

    print('Finished read path')
    return sample_paths

# read label
def read_label(path):
    with open(path, 'r') as label_fi:
        label_data = []
        for line in label_fi.readlines():
            if int(line.split('|')[0]) == 2:
                label = 0
            if int(line.split('|')[0]) == 1:
                label = 1
            label_data.append(label)
        label_data = np.asarray(label_data)

    print('Finished read label')
    return label_data

# xgboost needs data in a certain format:
# label 1:value1 2:value2 5:value5 ...
# load feature from fe_path and label from la_path, save the converted data into save_path
def save2xgdata(fe_path, la_path):
    label = read_label(la_path)
    # feature and label is in numpy.asarray format
    saver = open(fe_path+'.libsvm', 'w')
    with open(fe_path, 'r') as data_fi:
    	lines = data_fi.readlines()
    	dimension = int(lines[0].strip())
        count = 0
        for line in lines[1:]:
            #there is a '\n' at the last of the list
            data_index = [int(i) for i in line.split(';')[:-1][3::2]]
            data_value = [float(i) for i in line.split(';')[:-1][4::2]]
            data_str = str(label[count])
            count += 1
            try:
                for i in range(len(data_index)):
                    data_str += (' '+str(data_index[i])+':'+str(data_value[i]))
            except IndexError:
                pass
            # no need to designate dimension of the data
            if len(data_index)<dimension:
                data_str += (' '+str(dimension)+':'+str(0))
            saver.writelines(data_str+'\n')
    print('Finished read feature')

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def to_NN(data,label,path,NN_name='NN_train.txt',NN_label_name='NNAI_train.txt'):
    """
    将传入的 data label  path等等数据，存储为NN数据格式 
    :param data: ndarray (N,D)
    :param label: ndarray (N,)
    :param path: ndarray (N,)
    :return: 
    """
    assert data.shape[0] == label.shape[0] == path.shape[0]

    N = data.shape[0]
    dim = data.shape[1]

    with open(NN_name,'w') as NN:
        with open(NN_label_name,'w') as NN_label:
            NN.write(str(dim)+'\n')
            NN_label.write(str(dim)+'\n')
            for i in tqdm(range(N)):
                case = data[i]
                indice = np.nonzero(case)[0]
                line = str(label[i])+';'
                for index in indice:
                    line += str(index+1)+';'+str(case[index])+';'
                line_AI= str(label[i]) + '|' + path[i]
                if(i!=N-1):
                    line += '\n'
                    line_AI += '\n'
                NN.write(line)
                NN_label.write(line_AI)



if __name__ == '__main__':
    # labels,preds_score = get_predictedResults_from_file('Data/rslt_app.txt')
    # print labels,preds_score
    arg = arg_parser()
    save2xgdata(arg.sample,arg.label)