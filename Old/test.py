#coding=utf-8
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import make_scorer
from tools import get_csr_labels,tune_classifier
from xg_train import save2xgdata
import pprint
import pandas as pd
import time

def find_opt_param(param, params, dtrain, num_boost_round, watchlist, upper_bound, down_bound, stride=1):
    evals_result = {}
    results = []
    for value in range(upper_bound, down_bound, stride):
        print '=========== %s  :  %d  ===========' % (param, value)
        params[param] = value
        xgb.train(params, dtrain, num_boost_round, watchlist, evals_result=evals_result)
        results.append((value, evals_result['eval']['auc'][-1]))
    results.sort(key=lambda k: k[1])
    return results


def find_opt_floatparam(param, params, params_list, dtrain, num_boost_round, watchlist):
    evals_result = {}
    results = []
    for value in params_list:
        print '=========== %s  :  %f  ===========' % (param, value)
        params[param] = value
        xgb.train(params, dtrain, num_boost_round, watchlist, evals_result=evals_result)
        results.append((value, evals_result['eval']['auc'][-1]))
    results.sort(key=lambda k: k[1])
    return results
def tune_xgb(params_dict,is_print=True):
    global params
    global watchlist
    global dtrain
    global num_boost_round
    global evals_result
    results = {}
    temp_param = [None,0.0]
    for k,values in params_dict.items():
        results[k] = []
        temp_param = [None, 0.0]
        for v in values:
            print '=========== %s  :  %s  ===========' % (k, str(v))
            params[k] = v
            xgb.train(params, dtrain, num_boost_round, watchlist, evals_result=evals_result)
            eval_auc = evals_result['eval']['auc'][-1]
            if temp_param[1]<eval_auc:
                temp_param[0] = v
                temp_param[1] = eval_auc
            results[k].append((v, eval_auc))
        results[k].sort(key=lambda k: k[1])
        params[k] = temp_param[0]
        print '***** %s : %s : %s'%(k,str(temp_param[0]),str(temp_param[1]))
        if is_print == True:
            print 'result for %s',k
            pprint.pprint(results[k])
    return results
def print_params(params):
    pd_params = pd.DataFrame(params, index=['value'])
    print "==========  params   =========="
    print pd_params.T
def tune_xgb_cv(params_untuned,scoring='roc_auc', n_jobs=1, cv=5):
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

def set_params():
    global  params
    params['subsample'] = 1.0
    params['colsample_bytree'] = 0.8
    params['max_depth'] = 50
    params['min_child_weight'] = 0.1

def get_num_boost_round():
    global evals_result
    result = evals_result['train']['auc']
    result.sort()
    print result
    print result[-1]

def tune_num_boost_round():
    # global watchlist
    global num_boost_round
    global  evals_result
    evals_result = {}
    xgb.train(params=params_no_sklearn,dtrain=dtrain,num_boost_round=num_boost_round,evals=watchlist,evals_result=evals_result)
    evals_result = evals_result['eval']['map']
    pprint.pprint(evals_result)
    max = 0.0
    max_loc = 0
    for i,v in enumerate(evals_result):
        # print '%d ...... %d : %d'%(i,max_loc,max)
        if v>max:
            max = v
            max_loc = i
    print "max_loc : %d ,  max : %d"%(max_loc,max)
    num_boost_round = max_loc+1
    print '****  num_boost_round : ', num_boost_round

def set_params_no_sklearn():
    global params_no_sklearn
    params_no_sklearn['eta'] = 0.1
    params_no_sklearn['max_depth'] = params_sklearn['max_depth']
    params_no_sklearn['subsample'] = params_sklearn['subsample']
    params_no_sklearn['min_child_weight'] = params_sklearn['min_child_weight']
    params_no_sklearn['colsample_bytree'] = params_sklearn['colsample_bytree']


if __name__ == '__main__':
    # save2xgdata('Data/OSCE/NN_train.txt','Data/OSCE/NNAI_train.txt')
    # save2xgdata('Data/OSCE/NN_test.txt','Data/OSCE/NNAI_test.txt')
    # exit()
    xgmat_dataset_path = './Data/OSCE/NN_train.txt.xgmat'
    # xgb.train()
    x,y = get_csr_labels(xgmat_dataset_path=xgmat_dataset_path)
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val,y_val)
    dtrain_whole = xgb.DMatrix(xgmat_dataset_path)

    watchlist  = [(dtrain,'train'),(dval,'eval')]
    watchlist_whole  = [(dtrain_whole,'train')]
    evals_result = {}

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
        'scale_pos_weight':0.1,
        'refresh_leaf':1,

        'objective':'binary:logistic',
        'base_score':0.5,
        # 'eval_metric':'auc',
        'eval_metric':'map',
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
        'scale_pos_weight': 0.1,
        # 'refresh_leaf':1,

        'objective': 'binary:logistic',
        'base_score': 0.5,
        # 'eval_metric': 'auc',
        'seed': 0
    }
    # tune num_boost_round
    """
    num_boost_round = 10000
    tune_num_boost_round()
    params_sklearn['n_estimators'] = num_boost_round
    """
    params_sklearn['n_estimators'] = 594
    # num_boost_round = 594
    # eval_metric = 'auc'
    # early_stopping_rounds = 10
    # set_params()
    # xg_model = xgb.train(params,dtrain=dtrain_whole,num_boost_round=10,evals=watchlist_whole,evals_result=evals_result)
    # xg_model.save_model('Models/OSCE/tuned.xgmodel')
    # get_num_boost_round()
    # exit()
    # tune the parameters
    # params_max_depth = range(6,11,2)
    params_max_depth = range(50,61,10)
    params_sklearn['max_depth'] = 8
    params_subsample = [0.8,0.9,1.0]
    params_sklearn['subsample'] = 1.0
    # params_sklearn['colsample_bylevel'] = 1.0
    params_sklearn['min_child_weight'] = 0.03
    params_sklearn['colsample_bytree'] = 0.8
    # params_subsample = [0.8,1.0]
    # params_min_child_weight  = [0.02,0.03,0.04]
    # params_min_child_weight  = [0.5,1]
    # params_colsample_bytree = [0.7,0.8,0.9,1.0]
    # params_colsample_bytree = [0.7,0.8,0.9]
    params_untuned = [
                    # dict(max_depth=params_max_depth),
                    # dict(subsample=params_subsample),
                    # dict(min_child_weight=params_min_child_weight),
                    # dict(colsample_bytree=params_colsample_bytree),
                    ]
    # tune_xgb_cv(params_untuned,cv=5,scoring='average_precision',n_jobs=3)
    # print_params(params_sklearn)
    set_params_no_sklearn()
    print_params(params_no_sklearn)
    # tune the num_boost_round again
    num_boost_round = 7637
    # tune_num_boost_round()

    model = xgb.train(params_no_sklearn,dtrain=dtrain_whole,num_boost_round=num_boost_round,evals=watchlist_whole)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model.save_model('Models/OSCE/tuned_'+time_str+'.xgmodel')
    """
    params_tuned = dict(max_depth=params_max_depth,
                        subsample=params_subsample,
                        min_child_weight=params_min_child_weight,
                        colsample_bytree = params_colsample_bytree)
    """
    # print tune_xgb(params_tuned)





    """
    results = find_opt_floatparam('min_child_weight',params,[0.4,0.2,0.6],dtrain,num_boost_round,watchlist)

    # results = find_opt_param('min_child_weight',params,dtrain,num_boost_round,watchlist,1,10,4)
    print results
    print results[-1]
    """


