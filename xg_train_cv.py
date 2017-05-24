#coding=utf-8
"""
基于xgboost的cv函数进行调参
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from tools import get_csr_labels,save2xgdata
import pprint
import pandas as pd
import time
import numpy as np
import ConfigParser
import argparse
import os

from tools import get_csr_labels,save2xgdata

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    booster = cf.get('xg_conf', 'booster')
    silent = int(cf.get('xg_conf','silent'))
    nthread = int(cf.get('xg_conf', 'nthread'))

    eta = float(cf.get('xg_conf', 'eta'))
    gamma = float(cf.get('xg_conf', 'gamma'))
    max_delta_step = float(cf.get('xg_conf','max_delta_step'))
    p_lambda = float(cf.get('xg_conf', 'lambda'))
    alpha = float(cf.get('xg_conf', 'alpha'))
    sketch_eps = float(cf.get('xg_conf', 'sketch_eps'))
    refresh_leaf = int(cf.get('xg_conf', 'refresh_leaf'))

    objective = cf.get('xg_conf', 'objective')
    base_score = float(cf.get('xg_conf', 'base_score'))
    eval_metric = cf.get('xg_conf', 'eval_metric')
    ascend = int(cf.get('xg_conf','ascend'))
    seed = int(cf.get('xg_conf', 'seed'))

    num_round = int(cf.get('xg_conf', 'num_round'))
    save_period = int(cf.get('xg_conf', 'save_period'))
    eval = int(cf.get('xg_conf', 'eval'))
    cv = int(cf.get('xg_conf','cv'))

    t_num_round = int(cf.get('xg_conf','num_round'))
    t_max_depth = [int(i) for i in cf.get('xg_conf','max_depth').split(',')]
    t_subsample = [float(i) for i in cf.get('xg_conf','subsample').split(',')]
    t_min_child_weight = [float(i) for i in cf.get('xg_conf','min_child_weight').split(',')]
    t_colsample_bytree = [float(i) for i in cf.get('xg_conf','colsample_bytree').split(',')]

    t_param = {'num_round':t_num_round,'max_depth':t_max_depth,'subsample':t_subsample,
               'min_child_weight':t_min_child_weight,'colsample_bytree':t_colsample_bytree}

    params = {'booster': booster, 'objective': objective, 'silent': silent, 'eta': eta, 'gamma': gamma,
             'max_delta_step':max_delta_step,'lambda':p_lambda,'alpha':alpha,'sketch_eps':sketch_eps,
             'refresh_leaf':refresh_leaf,'base_score':base_score,'eval_metric':eval_metric,
             'seed':seed,'nthread': nthread}

    others = {'num_round':num_round,'cv':cv,'ascend':ascend}
    data = cf.get('xg_conf', 'data')

    if int(cf.get('xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
        try:
            label = cf.get('xg_conf', 'label')
            save2xgdata(data, label)
            data += '.libsvm'
        except:
            pass
    else:
        data = cf.get('xg_conf', 'xgdata')
    return data, params,t_param,others

def get_negative_positive_ratio(y):
    labels_np = np.array(y)
    neg_num = np.sum(labels_np==0)
    pos_num = np.sum(labels_np==1)
    return neg_num/pos_num

def tune_num_boost_round(params,dtrain,num_boost_round,watchlist,ascend=True):

    evals_result = {}
    xgb.train(params=params,dtrain=dtrain,num_boost_round=num_boost_round,evals=watchlist,evals_result=evals_result)
    evals_result = evals_result['eval'][params['eval_metric']]
    if(ascend==True):
        loc = max(enumerate(evals_result), key=lambda x: x[1])[0]
    else:
        loc = min(enumerate(evals_result), key=lambda x: x[1])[0]
    loc += 1
    print('****  num_boost_round : %s'%loc)
    return loc

if __name__ == '__main__':
    arg = arg_parser()
    xgdata,params,params_t,params_other = conf_parser(arg.conf)

    x, y = get_csr_labels(xgdata)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, y_val)
    dtrain_whole = xgb.DMatrix(xgdata)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    watchlist_whole = [(dtrain_whole, 'eval')]

    scale_pos_weight = get_negative_positive_ratio(y)
    params['scale_pos_weight'] = scale_pos_weight
    # tune the parameter num_round
    num_round = tune_num_boost_round(params,dtrain,params_other['num_round'],watchlist,ascend=params_other['ascend'])

    params_t = [dict(max_depth=params_t['max_depth']),
                dict(subsample=params_t['subsample']),
                dict(min_child_weight=params_t['min_child_weight']),
                dict(colsample_bytree=params_t['colsample_bytree'])]
    for param_t in params_t:
        k = param_t.keys()[0]
        values = param_t[k]
        if(k=='num_round'):
            continue
        # pprint.pprint(params)
        print('========== ',k,' ========== ',values)
        result = []
        if(len(values) == 1):
            params[k] = values[0]
            continue
        for v in values:
            print('**** for : %s ****\n'%(str(v)))
            params[k] = v
            result_df = xgb.cv(params=params,
                               dtrain=dtrain_whole,
                               num_boost_round=num_round,
                               nfold=params_other['cv'],
                               metrics=params['eval_metric'],
                               verbose_eval=False,
                               show_stdv=False,
                               shuffle=True)
            # print(result_df)
            assert result_df.columns[0]=='test-error-mean','choose the correct column\n'
            result_np = result_df.as_matrix()
            result.append(float(result_np[-1][0]))
        print(zip(values,result))
        if(params_other['ascend'] == 1):
            loc = max(enumerate(result),key=lambda x:x[1])[0]
        else:
            loc = min(enumerate(result),key=lambda x:x[1])[0]
        params[k] = values[loc]
        print('%s : %s\n'%(k,params[k]))
    num_round = tune_num_boost_round(params,dtrain_whole,params_other['num_round'],watchlist_whole,ascend=params_other['ascend'])
    model = xgb.train(params,dtrain_whole,num_round,watchlist_whole)
    pprint.pprint(params)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    model.save_model(arg.output + '/' + time_str + '.xgmodel')
    print('saved : %s' % (arg.output + '/' + time_str + '.xgmodel'))
