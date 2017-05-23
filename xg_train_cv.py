#coding=utf-8
"""
基于xgboost的cv函数进行调参
"""
import xgboost as xgb

if __name__ == '__main__':
    data_path = './Data/features/train.svmlib'
    dtrain_whole = xgb.DMatrix(data_path)

    params = {
        'booster': 'gbtree',
        'silent': 1,
        'nthread':1,

        'eta': 0.1,
        'gamma': 0.2,
        'max_depth': 8,
        'min_child_weight': 5,
        'max_delta_step': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'lambda': 300,
        'alpha': 0,
        'sketch_eps': 0.03,
        'scale_pos_weight': 900,
        'refresh_leaf': 1,

        'objective': 'binary:logistic',
        'base_score': 0.5,
        'eval_metric':'error',
        'seed': 0
    }

    result = xgb.cv(params=params,dtrain=dtrain_whole,num_boost_round=10,
                    nfold=5,metrics='error',shuffle=True)
    print(result)