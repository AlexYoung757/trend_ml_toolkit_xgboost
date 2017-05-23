#coding=utf-8
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

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=True)
    # parser.add_argument('-v', '--visual', default=30)
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
    max_depth = int(cf.get('xg_conf', 'max_depth'))
    min_child_weight = float(cf.get('xg_conf', 'min_child_weight'))
    max_delta_step = float(cf.get('xg_conf','max_delta_step'))
    subsample = float(cf.get('xg_conf', 'subsample'))
    p_lambda = float(cf.get('xg_conf', 'lambda'))
    alpha = float(cf.get('xg_conf', 'alpha'))
    sketch_eps = float(cf.get('xg_conf', 'sketch_eps'))
    scale_pos_weight = float(cf.get('xg_conf', 'scale_pos_weight'))
    refresh_leaf = int(cf.get('xg_conf', 'refresh_leaf'))

    objective = cf.get('xg_conf', 'objective')
    base_score = float(cf.get('xg_conf', 'base_score'))
    eval_metric = cf.get('xg_conf', 'eval_metric')
    seed = int(cf.get('xg_conf', 'seed'))

    num_round = int(cf.get('xg_conf', 'num_round'))
    save_period = int(cf.get('xg_conf', 'save_period'))
    eval = int(cf.get('xg_conf', 'eval'))
    # xgdata = cf.get('xg_conf', 'xgdata')
    eval_metric_sk = cf.get('xg_conf', 'eval_metric_sk')
    cv = int(cf.get('xg_conf','cv'))
    n_jobs = int(cf.get('xg_conf','n_jobs'))

    t_num_round = int(cf.get('xg_conf','t_num_round'))
    t_max_depth = [int(i) for i in cf.get('xg_conf','t_max_depth').split(',')]
    t_subsample = [float(i) for i in cf.get('xg_conf','t_subsample').split(',')]
    t_min_child_weight = [float(i) for i in cf.get('xg_conf','t_min_child_weight').split(',')]
    t_colsample_bytree = [float(i) for i in cf.get('xg_conf','t_colsample_bytree').split(',')]

    t_param = {'t_num_round':t_num_round,'t_max_depth':t_max_depth,'t_subsample':t_subsample,
               't_min_child_weight':t_min_child_weight,'t_colsample_bytree':t_colsample_bytree}

    param = {'booster': booster, 'objective': objective, 'silent': silent, 'eta': eta, 'gamma': gamma,
             'min_child_weight': min_child_weight,'max_delta_step':max_delta_step,'subsample':subsample,
             'lambda':p_lambda,'alpha':alpha,'sketch_eps':sketch_eps,'scale_pos_weight':scale_pos_weight,
             'refresh_leaf':refresh_leaf,'base_score':base_score,'eval_metric':eval_metric,
             'seed':seed,'max_depth': max_depth,'nthread': nthread}

    param_sk = {'n_estimators':num_round,'objective': objective, 'silent': silent, 'learning_rate': eta, 'gamma': gamma,
             'min_child_weight': min_child_weight,'max_delta_step':max_delta_step,'subsample':subsample,
             'reg_lambda':p_lambda,'reg_alpha':alpha,'scale_pos_weight':scale_pos_weight,'base_score':base_score,
             'seed':seed,'max_depth': max_depth,'nthread': nthread}
    others = {'num_round':num_round,'eval_metric_sk':eval_metric_sk,'cv':cv,'n_jobs':n_jobs}
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
    return data, param,param_sk,others,t_param

def print_params(params):
    pd_params = pd.DataFrame(params, index=['value'])
    print "==========  params   =========="
    print pd_params.T

def tune_xgb_cv(params_untuned,scoring='roc_auc', n_jobs=4, cv=5):
    # global  dtrain_whole
    global  num_boost_round
    global  params_sklearn

    # global x
    # global y
    for param_untuned in params_untuned:
        print '==========  ', param_untuned, '  =============='
        print_params(params_sklearn)
        estimator = xgb.XGBClassifier(**params_sklearn)
        grid_search = GridSearchCV(estimator, param_grid=param_untuned, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=10)
        grid_search.fit(x, y)
        df0 = pd.DataFrame(grid_search.cv_results_)
        df = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_train_score', 'mean_test_score']]
        # print df0
        print df
        print 'the best_params : ', grid_search.best_params_
        print 'the best_score  : ', grid_search.best_score_
        # print grid_search.cv_results_
        for k,v in grid_search.best_params_.items():
            params_sklearn[k] = v
            if len(params_untuned)==1:
                return v


def set_params():
    global  params
    # params['subsample'] = 1.0
    params['colsample_bytree'] = 0.8
    params['max_depth'] = 8
    # params['min_child_weight'] = 0.1

def get_num_boost_round():
    global evals_result
    result = evals_result['train']['auc']
    result.sort()
    print result
    print result[-1]

def tune_num_boost_round():
    # global watchlist
    global num_boost_round
    global evals_result
    global eval_metric_xgb_format
    evals_result = {}
    xgb.train(params=params_no_sklearn,dtrain=dtrain,num_boost_round=num_boost_round,evals=watchlist,evals_result=evals_result)
    evals_result = evals_result['eval'][eval_metric_xgb_format]
    # pprint.pprint(evals_result)
    max = 0.0
    max_loc = 0
    for i,v in enumerate(evals_result):
        # print '%d ...... %d : %d'%(i,max_loc,max)
        if v>max:
            max = v
            max_loc = i
    # print "max_loc : %s ,  max : %s"%(max_loc,max)
    num_boost_round = max_loc+1
    print('****  num_boost_round : ', num_boost_round)

def set_params_no_sklearn():
    global params_no_sklearn
    # params_no_sklearn['eta'] = 0.05
    params_no_sklearn['max_depth'] = params_sklearn['max_depth']
    params_no_sklearn['subsample'] = params_sklearn['subsample']
    params_no_sklearn['min_child_weight'] = params_sklearn['min_child_weight']
    params_no_sklearn['colsample_bytree'] = params_sklearn['colsample_bytree']

def my_range(begin,end,step):
    result = []
    result.append(begin)
    temp = begin + step
    while temp < end:
        result.append(temp)
        temp = temp + step
    return result
def get_negative_positive_ratio():
    labels_np = np.array(y)
    neg_num = np.sum(labels_np==0)
    pos_num = np.sum(labels_np==1)
    return neg_num/pos_num

if __name__ == '__main__':
    # save2xgdata('Data/OC-vuq2/NN_train.txt','Data/OC-vuq2/NNAI_train.txt')
    # save2xgdata('Data/OC-vuq2/NN_test.txt','Data/OC-vuq2/NNAI_test.txt')
    # exit()
    arg = arg_parser()
    xgmat_dataset_path,params_no_sklearn,params_sklearn,others,t_params = conf_parser(arg.conf)
    pprint.pprint(t_params)
    # xgmat_dataset_path = './Data/OC-vuq1/NN_train.txt.xgmat'
    # xgb.train()
    x,y = get_csr_labels(xgmat_dataset_path=xgmat_dataset_path)
    scale_pos_weight = get_negative_positive_ratio()
    params_no_sklearn['scale_pos_weight'] = scale_pos_weight
    params_sklearn['scale_pos_weight'] = scale_pos_weight
    # print get_negative_positive_ratio()
    # exit()
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val,y_val)
    dtrain_whole = xgb.DMatrix(xgmat_dataset_path)

    eval_metric_xgb_format = params_no_sklearn['eval_metric']
    eval_metric_sklearn_format = others['eval_metric_sk']

    watchlist = [(dtrain,'train'),(dval,'eval')]
    watchlist_whole  = [(dtrain_whole,'train')]
    evals_result = {}
    '''
    params_no_sklearn = {
        'booster':'gbtree',
        'silent':1,
        'nthread':4,

        'eta':0.1,
        'gamma':0.2,
        'max_depth':10,
        'min_child_weight':5,
        'max_delta_step':0,
        'subsample':1,
        'colsample_bytree':1,
        'colsample_bylevel':1,
        'lambda':300,
        'alpha':0,
        'sketch_eps':0.03,
        'scale_pos_weight':900,
        'refresh_leaf':1,

        'objective':'binary:logistic',
        'base_score':0.5,
        'eval_metric':eval_metric_xgb_format,
        'seed':0
    }

    params_sklearn = {
        # 'booster': 'gbtree',
        'silent': 1,
        'nthread': 4,

        # 'eta': 0.1,
        'learning_rate':0.1,
        'n_estimators':5,
        'gamma': 0.2,
        'max_depth': 10,
        'min_child_weight': 5,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        # 'lambda ': 300,
        'reg_lambda':300,
        # 'alpha ': 0,
        'reg_alpha':0,
        # 'sketch_eps': 0.03,
        'scale_pos_weight': 900,
        # 'refresh_leaf':1,

        'objective': 'binary:logistic',
        'base_score': 0.5,
        # 'eval_metric': 'auc',
        'seed': 0
    }
    '''
    # tune num_boost_round
    num_boost_round = t_params['t_num_round']
    tune_num_boost_round()
    params_sklearn['n_estimators'] = num_boost_round

    # eval_metric = 'auc'
    # early_stopping_rounds = 10
    # set_params()
    # xg_model = xgb.train(params,dtrain=dtrain_whole,num_boost_round=10,evals=watchlist_whole,evals_result=evals_result)
    # xg_model.save_model('Models/OSCE/tuned.xgmodel')
    # get_num_boost_round()
    # exit()

    # tune the parameter max_depth
    # params_sklearn['max_depth'] = 8
    # params_max_depth = range(10,201,10)

    # params_max_depth = [6,8,10]
    # temp = tune_xgb_cv([dict(max_depth=params_max_depth)],scoring=eval_metric_sklearn_format)
    # if temp!=10 or temp!=200:
    #     params_max_depth = range(temp-9,temp+10,1)
    #     temp = tune_xgb_cv([dict(max_depth=params_max_depth)], scoring=eval_metric_sklearn_format)
    # print("***  params_max_depth : %d")%(temp)

    # params_max_depth = range(50,61,10)

    # params_sklearn['subsample'] = 0.9
    # params_subsample = [ 0.8, 0.9, 1.0]
    # temp = tune_xgb_cv([dict(subsample=params_subsample)], scoring=eval_metric_sklearn_format)
    # print("***  params_subsample : %s") % (temp)
    # params_subsample = [0.8,1.0]

    # params_sklearn['subsample'] = 1.0
    # params_sklearn['min_child_weight']=0.6
    # params_min_child_weight = [0.4,0.6,0.8]
    # temp = tune_xgb_cv([dict(min_child_weight=params_min_child_weight)], scoring=eval_metric_sklearn_format)

    """
    params_min_child_weight = my_range(0.01,2.02,0.5)
    temp = tune_xgb_cv([dict(min_child_weight=params_min_child_weight)], scoring=eval_metric_sklearn_format)
    if temp!=0.01 or temp!=2.02:
        params_min_child_weight = my_range(temp-0.4,temp+0.5,0.1)
        temp = tune_xgb_cv([dict(min_child_weight=params_min_child_weight)], scoring=eval_metric_sklearn_format)
    print("***  params_min_child_weight : %s") % (temp)
    # params_min_child_weight  = [0.5,1]
    """

    # params_colsample_bytree = [0.8,0.9,1.0]
    # temp = tune_xgb_cv([dict(colsample_bytree=params_colsample_bytree)], scoring=eval_metric_sklearn_format)
    # print("***  params_colsample_bytree : %s") % (temp)
    # params_colsample_bytree = [0.7,1.0]


    params_untuned = [
                    dict(max_depth=t_params['t_max_depth']),
                    dict(subsample=t_params['t_subsample']),
                    dict(min_child_weight=t_params['t_min_child_weight']),
                    dict(colsample_bytree=t_params['t_colsample_bytree']),
                    ]
    tune_xgb_cv(params_untuned,scoring=eval_metric_sklearn_format,
                n_jobs=others['n_jobs'],cv=others['cv'])

    print_params(params_sklearn)
    set_params_no_sklearn()
    print_params(params_no_sklearn)
    # tune the num_boost_round again
    tune_num_boost_round()

    model = xgb.train(params_no_sklearn,dtrain=dtrain_whole,num_boost_round=num_boost_round,evals=watchlist_whole)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model.save_model(arg.output+'/'+time_str+'.xgmodel')
    print('saved : %s'%(arg.output+'/'+time_str+'.xgmodel'))

