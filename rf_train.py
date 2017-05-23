#coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from tools import get_csr_labels
from sklearn.model_selection import train_test_split,GridSearchCV,ParameterGrid
from sklearn.metrics import *
from estimation import compare_model
import pandas as pd

def val_tune_rf(estimator,x_train,y_train,x_val,y_val,params):
    params_list = list(ParameterGrid(params))
    print params_list
    print y_val
    results = []
    for param in params_list:
        print '=========  ',param
        estimator.set_params(**param)
        estimator.fit(x_train,y_train)
        preds_prob = estimator.predict_proba(x_val)
        # print preds_prob[:,1]
        result = roc_auc_score(y_val,preds_prob[:,1])
        print 'roc_auc_score : %f'%result
        results.append((param,result))
    results.sort(key=lambda k: k[1])
    print results
    print results[-1]

def tune_n_estimators_cv(estimator,params,X_train,Y_train):
    grid_search = GridSearchCV(estimator,param_grid=params,scoring='roc_auc',n_jobs=-1,cv=10,verbose=10)
    grid_search.fit(X_train,Y_train)
    return grid_search.best_params_

def tune_max_depth_and_min_samples_split_cv(estimator,params,X_train,Y_train):
    grid_search = GridSearchCV(estimator,param_grid=params,scoring='roc_auc',n_jobs=-1,cv=10,verbose=10)
    grid_search.fit(X_train,Y_train)
    return grid_search.best_params_

def tune_min_samples_split_and_min_samples_leaf_cv(estimator,params,X_train,Y_train):
    grid_search = GridSearchCV(estimator,param_grid=params,scoring='roc_auc',n_jobs=-1,cv=10,verbose=10)
    grid_search.fit(X_train,Y_train)
    return grid_search.best_params_

def tune_rf(estimator,params,X_train,Y_train):
    print '==========  ',params,'  =============='
    grid_search = GridSearchCV(estimator,param_grid=params,scoring='roc_auc',n_jobs=3,cv=5,verbose=5)
    grid_search.fit(X_train,Y_train)
    df0 = pd.DataFrame(grid_search.cv_results_)
    df = pd.DataFrame(grid_search.cv_results_)[['params','mean_train_score','mean_test_score']]
    # print df0
    print df
    print 'the best_params : ',grid_search.best_params_
    print 'the best_score  : ',grid_search.best_score_
    # print grid_search.cv_results_
    return grid_search.best_params_

if __name__ == '__main__':

    # 加载数据集
    train_dataset_path = './Data/NN_train.txt.xgmat'
    test_dataset_path = './Data/NN_test.txt.xgmat'
    x, y = get_csr_labels(xgmat_dataset_path=train_dataset_path)
    x_test,y_test = get_csr_labels(xgmat_dataset_path=test_dataset_path)
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 初始化模型
    rf_model=RandomForestClassifier(max_features="log2",verbose=5)

    # 调整参数 n_estimators
    params_n_estimators0 = dict(n_estimators=range(100,1001,100))
    param_n_estimators0 = tune_rf(rf_model,params_n_estimators0,x,y)
    params_n_estimators1 = dict(n_estimators=range(param_n_estimators0['n_estimators']-100,param_n_estimators0['n_estimators']+101,10))
    param_n_estimators1 = tune_rf(rf_model,params_n_estimators1,x,y)
    params_n_estimators2 = dict(n_estimators=range(param_n_estimators1['n_estimators']-10,param_n_estimators1['n_estimators']+11,1))
    rf_model.set_params(**params_n_estimators2)

    # 调整参数 max_depth 和 min_samples_split
    params_max_depth_and_min_samples_split = dict(max_depth=range(100,102,1),min_samples_split=range(50,52,1))
    param_max_depth_and_min_samples_split = tune_rf(rf_model,params_max_depth_and_min_samples_split,x,y)
    rf_model.set_params(**param_max_depth_and_min_samples_split)

    # 调整参数 min_samples_split 和 min_samples_leaf
    params_min_samples_split_and_min_samples_leaf = dict(min_samples_split=range(100,130,20),min_samples_leaf=range(40,60,10))
    param_min_samples_split_and_min_samples_leaf = tune_rf(rf_model,params_min_samples_split_and_min_samples_leaf,x,y)
    rf_model.set_params(**param_min_samples_split_and_min_samples_leaf)

    # 模型评估
    rf_model_untuned = RandomForestClassifier()
    rf_model.fit(x,y)
    rf_model_untuned.fit(x,y)
    preds_prob_untuned = rf_model.predict_proba(x_test)[:,1]
    preds_prob = rf_model.predict_proba(x_test)[:,1]

    compare_model(y_test,preds_prob_untuned,preds_prob,"rf_model_untuned","rf_model_tuned")

    """
    rf_model = RandomForestClassifier(
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None)
    rf_model = RandomForestClassifier()
    params = dict(
                 n_estimators=[10,20,30],
                 # criterion="gini",
                 # max_depth=None,
                 # min_samples_split=2,
                 # min_samples_leaf=1,
                 # min_weight_fraction_leaf=0.,
                 # max_features="auto",
                 # max_leaf_nodes=None,
                 # min_impurity_split=1e-7,
                 # bootstrap=True,
                 # oob_score=False,
                 # random_state=None,
                 # verbose=0,
                 # warm_start=False,
                 # class_weight=None
    )
    params0 = dict(n_estimators=[100,150,200])
    grid_search = GridSearchCV(RandomForestClassifier(),param_grid=params0,n_jobs=4,scoring='roc_auc',cv=5,verbose=1)
    grid_search.fit(x,y)
    print grid_search.cv_results_
    print pd.DataFrame(grid_search.cv_results_)
    print grid_search.best_params_
    print grid_search.best_score_
    """
