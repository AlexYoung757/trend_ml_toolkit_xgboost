# coding=utf-8
import xgboost as xgb
import numpy as np
import pandas as pd


from tools import read_sample_path

def get_config():
    config = dict()
    config['model_path'] = './Models/OC-vuq1/2017_05_25_12_52_34.xgmodel'
    config['data_path'] = './Data/OC-vuq1/NN_test.txt.libsvm'
    config['result_path'] = './Output/result.csv'
    config['label_path'] = './Data/OC-vuq1/NNAI_test.txt'
    return config

if __name__ == '__main__':
    config = get_config()

    dtest = xgb.DMatrix(config['data_path'])
    labels = dtest.get_label()
    labels = labels.astype(int)
    model = xgb.Booster(model_file=config['model_path'])
    labels_pred = model.predict(dtest)

    labels = np.array(labels)[...,np.newaxis]
    labels_pred = np.array(labels_pred)[...,np.newaxis]
    label_pathes = read_sample_path(config['label_path'])[...,np.newaxis]

    assert len(labels) == len(labels_pred) ==  len(label_pathes)

    result_np = np.concatenate((label_pathes,labels_pred),axis=1)
    result_np = np.concatenate((result_np,labels),axis=1)

    result_df = pd.DataFrame(result_np)
    print(result_df)

    if(config['result_path']!=''):
        result_df.to_csv(config['result_path'],index=False)



