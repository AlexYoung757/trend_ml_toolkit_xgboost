import numpy as np
import xgboost as xgb
import ConfigParser
import argparse
import random
import os
import xg_plot
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-k','--nfold', default=5)
    parser.add_argument('-s','--shuffle', default=1)
    return parser.parse_args()

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
    feature = list()
    saver = open(fe_path+'.xgmat', 'w')
    with open(fe_path, 'r') as data_fi:
    	lines = data_fi.readlines()
    	dimension = int(lines[0].strip())
        count = 0
        for line in lines[1:]:
            data_index = [int(i) for i in line.split(';')[:-1][3::2]]
            data_value = [float(i) for i in line.split(';')[:-1][4::2]]
            data_str = str(label[count])
            count += 1
            try:
                for i in range(len(data_index)):
                    data_str += (' '+str(data_index[i])+':'+str(data_value[i]))
            except IndexError:
                pass
            if len(data_index)<dimension: # Ensure training data and test data keeping the same dimension
                data_str += (' '+str(dimension-1)+':'+str(0))
            saver.writelines(data_str+'\n')
    print('Finished read feature')
    positive_size = sum(label)
    negative_size = len(label) - sum(label)
    return positive_size, negative_size

# Give preds and true labels, output performance indices including Accuracy, Precision, Recall
def evaluation(preds, label):
    preds = [int(preds[i]>=0.5) for i in range(len(preds))]
    accuracy = sum([int(preds[i]==label[i]) for i in range(len(preds))]) / (len(preds)+0.0)
    precision = sum([int(preds[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(preds)+0.0)
    recall = sum([int(label[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(label)+0.0)
    return accuracy, precision, recall

def select(List, index):
    selected = [List[i] for i in index]
    return selected

# Do split
def xgb_cv(data, positives, negatives, param, nfold, shuffle):
    pos_index = range(positives); neg_index = range(negatives)
    if shuffle:
        print'Shuffling Data'; random.shuffle(pos_index); random.shuffle(neg_index)
    pos_step = int(positives/nfold)
    neg_step = int(negatives/nfold)
    train_write = open('tmp_train.xgmat', 'w')
    test_write = open('tmp_test.xgmat', 'w')
    accuracy = np.zeros(nfold); precision = np.zeros(nfold); recall = np.zeros(nfold)
    with open(data) as file:
        lines = file.readlines()
        for i in range(nfold):
            print('\nRound %d'%i)
            pos = np.ones(positives)
            neg = np.ones(negatives)
            pos[i*pos_step:(i+1)*pos_step] = 0; neg[i*neg_step:(i+1)*neg_step] = 0
            train_str = select(lines, np.where(np.concatenate((pos, neg))==1)[0])
            test_str = select(lines, np.where(np.concatenate((pos, neg)) == 0)[0])
            train_write.writelines(train_str); train_write.flush()
            test_write.writelines(test_str); test_write.flush()
            train = xgb.DMatrix('tmp_train.xgmat'); test = xgb.DMatrix('tmp_test.xgmat')
            if param['eval']:
                watchlist = [(train, 'train')]
                xg_model = xgb.train(param, train, param['num_round'], watchlist)
            else:
                xg_model = xgb.train(param, train, param['num_round'])
            preds = xg_model.predict(test)
            accuracy[i], precision[i], recall[i] = evaluation(preds, test.get_label())
    return accuracy, precision, recall

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)
    booster = cf.get('xg_conf', 'booster')
    objective = cf.get('xg_conf', 'objective')
    silent = int(cf.get('xg_conf','silent'))
    eta = float(cf.get('xg_conf', 'eta'))
    gamma = float(cf.get('xg_conf', 'gamma'))
    min_child_weight = float(cf.get('xg_conf', 'min_child_weight'))
    max_depth = int(cf.get('xg_conf', 'max_depth'))
    num_round = int(cf.get('xg_conf', 'num_round'))
    save_period = int(cf.get('xg_conf', 'save_period'))
    eval = int(cf.get('xg_conf', 'eval'))
    nthread = int(cf.get('xg_conf', 'nthread'))
    param = {'booster':booster, 'objective':objective, 'silent':silent, 'eta':eta, 'gamma':gamma, 'min_child_weight':min_child_weight,
             'max_depth':max_depth, 'num_round':num_round, 'save_period':save_period, 'nthread':nthread, 'eval':eval}
    data = cf.get('xg_conf', 'data')
    if int(cf.get('xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
        try:
            label = cf.get('xg_conf', 'label')
            positives, negatives = save2xgdata(data, label)
            data += '.xgmat'
        except:
            pass
    return data, positives, negatives, param


if __name__ == '__main__':
    arg = arg_parser()
    data, positives, negatives, param = conf_parser(arg.conf)
    nfold = int(arg.nfold); shuffle = int(arg.shuffle)
    accuracy, precision, recall = xgb_cv(data, positives, negatives, param, nfold, shuffle)
    print('\nAccuracy: %.4f +- %.4f\nPrecision: %.4f +- %.4f\nRecall: %.4f +- %.4f'%(np.mean(accuracy), np.std(accuracy),
                                                                                  np.mean(precision), np.std(precision),
                                                                                  np.mean(recall), np.std(recall)))
    os.remove('tmp_train.xgmat')
    os.remove('tmp_test.xgmat')



