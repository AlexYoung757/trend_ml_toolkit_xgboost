#coding=utf-8
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pprint
import pandas as pd
import time
import numpy as np
import ConfigParser
import argparse
import pickle
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

    learning_rate = float(cf.get('xg_conf', 'learning_rate'))
    silent = int(cf.get('xg_conf', 'silent'))
    objective = cf.get('xg_conf', 'objective')
    gamma = float(cf.get('xg_conf', 'gamma'))
    max_delta_step = float(cf.get('xg_conf','max_delta_step'))
    colsample_bylevel = float(cf.get('xg_conf','colsample_bylevel'))
    reg_alpha = float(cf.get('xg_conf','reg_alpha'))
    reg_lambda = float(cf.get('xg_conf','reg_lambda'))
    base_score = float(cf.get('xg_conf','base_score'))

    scoring = cf.get('xg_conf','scoring')
    n_jobs = int(cf.get('xg_conf', 'n_jobs'))
    cv = int(cf.get('xg_conf','cv'))
    verbose = int(cf.get('xg_conf','verbose'))

    t_n_estimators = [int(i) for i in cf.get('xg_conf','n_estimators').split(',')]
    t_max_depth = [int(i) for i in cf.get('xg_conf','max_depth').split(',')]
    t_subsample = [float(i) for i in cf.get('xg_conf','subsample').split(',')]
    t_min_child_weight = [float(i) for i in cf.get('xg_conf','min_child_weight').split(',')]
    t_colsample_bytree = [float(i) for i in cf.get('xg_conf','colsample_bytree').split(',')]

    t_param = {'n_estimators':t_n_estimators,'max_depth':t_max_depth,'subsample':t_subsample,
               'min_child_weight':t_min_child_weight,'colsample_bytree':t_colsample_bytree
               }

    param_sk = {'objective': objective, 'silent': silent, 'learning_rate': learning_rate, 'gamma': gamma,
                'max_delta_step':max_delta_step,'reg_lambda':reg_lambda,'reg_alpha':reg_alpha,'base_score':base_score,
                'colsample_bylevel':colsample_bylevel
                }
    others = {'scoring':scoring,'cv':cv,'n_jobs':n_jobs,'verbose':verbose}

    data = cf.get('xg_conf', 'data')

    if int(cf.get('xg_conf','xgmat'))==0: # if it is not a xgmat file, than convert it
        try:
            label = cf.get('xg_conf', 'label')
            save2xgdata(data, label)
            data += '.xgmat'
        except:
            pass
    else:
        data = cf.get('xg_conf', 'xgdata')
    return data,param_sk,others,t_param

def print_params(params):
    pd_params = pd.DataFrame(params, index=['value'])
    print "==========  params   =========="
    print pd_params.T

def tune_xgb_cv(params_untuned,params_sklearn,scoring='roc_auc', n_jobs=4, cv=5,verbose=10):

    for param_untuned in params_untuned:
        print '==========  ', param_untuned, '  =============='
        print_params(params_sklearn)
        estimator = xgb.XGBClassifier(**params_sklearn)
        # if(param_untuned.keys()[0] == 'n_estimators'):
        #     cv = 1
        grid_search = GridSearchCV(estimator, param_grid=param_untuned, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=verbose)
        grid_search.fit(x, y)
        df = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_train_score', 'mean_test_score']]
        print df
        print 'the best_params : ', grid_search.best_params_
        print 'the best_score  : ', grid_search.best_score_
        for k,v in grid_search.best_params_.items():
            params_sklearn[k] = v
    return estimator,params_sklearn

def get_negative_positive_ratio(y):
    labels_np = np.array(y)
    neg_num = np.sum(labels_np==0)
    pos_num = np.sum(labels_np==1)
    return neg_num/pos_num

def set_params_untuned(t_params,params_sklearn):
    params_untuned = []
    for k,v in t_params.items():
        if(len(v) == 1):
            params_sklearn[k] = v[0]
        else:
            params_untuned.append({k:v})
    return params_untuned,params_sklearn

if __name__ == '__main__':

    arg = arg_parser()
    xgmat_dataset_path,params_sklearn,others,t_params = conf_parser(arg.conf)
    pprint.pprint(t_params)
    x,y = get_csr_labels(xgmat_dataset_path=xgmat_dataset_path)
    scale_pos_weight = get_negative_positive_ratio(y)
    params_sklearn['scale_pos_weight'] = scale_pos_weight

    params_untuned, params_sklearn = set_params_untuned(t_params,params_sklearn)
    print(params_untuned,params_sklearn)
    """
    params_untuned = [
                    dict(n_estimators=t_params['n_estimators']),
                    dict(max_depth=t_params['max_depth']),
                    dict(subsample=t_params['subsample']),
                    dict(min_child_weight=t_params['min_child_weight']),
                    dict(colsample_bytree=t_params['colsample_bytree']),
                    ]
    """
    if(len(params_untuned)!=0):
        classifier,params_sklearn = tune_xgb_cv(params_untuned,params_sklearn,
                                                scoring=others['scoring'],
                                                n_jobs=others['n_jobs'],
                                                cv=others['cv'],
                                                verbose=others['verbose'])
    else:
        classifier = XGBClassifier(**params_sklearn)
    print_params(params_sklearn)
    classifier.fit(x,y)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if(os.path.exists(arg.output) ==  False):
        os.makedirs(arg.output)
    f = open(arg.output+'/'+time_str+'.xgmodel','w')
    pickle.dump(classifier,f)
    print('saved : %s'%(arg.output+'/'+time_str+'.xgmodel'))

