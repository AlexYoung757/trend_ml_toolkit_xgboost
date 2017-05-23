#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from temp import get_csr_labels,tune_n_estimators

xgmat_dataset_path = './Data/NN_train.txt.xgmat'
x_train,y_train = get_csr_labels(xgmat_dataset_path=xgmat_dataset_path)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtrain_whole = xgb.DMatrix(xgmat_dataset_path)
"""
print "Step 1: Fix learning rate and number of estimators for tuning tree-based parameters\n"

xgb1 = XGBClassifier(
 learning_rate =0.3,
 n_estimators=100,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
n_estimators = tune_n_estimators(xgb1,dtrain,useTrainCV=True,cv_folds=10,early_stopping_rounds=10)
print '*****************    n_estimators  : %d'%n_estimators
exit()
"""
param_test1 = {
 'max_depth':range(11,12,1),
 'min_child_weight':range(1,7,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =1, n_estimators=10, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
print '=======================\n'
gsearch1.fit(x_train,y_train)
grid_scores = pd.DataFrame(gsearch1.grid_scores_).drop('cv_validation_scores',axis=1)
print grid_scores
print gsearch1.best_params_
print gsearch1.best_score_