import numpy as np
import xgboost as xgb
import ConfigParser
import cPickle as pickle
import argparse
import xg_plot
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-v', '--visual', default=30)
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
    saver = open(fe_path+'.xgmat', 'w')
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
    param = {'booster': booster, 'objective': objective, 'silent': silent, 'eta': eta, 'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_depth': max_depth, 'num_round': num_round, 'save_period': save_period, 'eval': eval,
             'nthread': nthread}
    data = cf.get('xg_conf', 'data')

    if int(cf.get('xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
        try:
            label = cf.get('xg_conf', 'label')
            save2xgdata(data, label)
            data += '.xgmat'
        except:
            pass
    return data, param

# Give train data and parameters, return a trained model
def xgb_train(xg_train, param):
    if param['eval']:
        watchlist=[(xg_train, 'train')]
        xg_model = xgb.train(param, xg_train, param['num_round'], watchlist)
    else:
        xg_model = xgb.train(param, xg_train, param['num_round'])
    return xg_model

if __name__ == '__main__':
    arg = arg_parser()
    xg_train, param = conf_parser(arg.conf)
    # print xg_train,param
    xg_model = xgb_train(xgb.DMatrix(xg_train), param); #print sorted(xg_model.get_fscore().items(), key=itemgetter(1), reverse=True)
    data = load_svmlight_file(xg_train) # for plot
    xg_plot.fea_plot(xg_model, data[0].toarray(), data[1], max_num_features=int(arg.visual))
    xg_plot.tsne_plot(xg_model, data[0].toarray(), data[1], max_num_features=int(arg.visual))
    plt.show()
    saver = file(arg.output, 'wb')
    pickle.dump(xg_model, saver)



