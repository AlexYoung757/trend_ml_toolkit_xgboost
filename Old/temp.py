import numpy as np
import scipy.sparse
import xgboost as xgb
from sklearn.metrics import *
from xg_predict_comp import compare_models
import pprint
from xg_train import save2xgdata




def get_features(fe_path):
    with open(fe_path, 'r') as data_fi:
        lines = data_fi.readlines()
        dimension = int(lines[0].strip())
        num_row = len(lines)-1
        print num_row,dimension
        features = np.zeros((num_row,dimension))
        row_index = 0
        for line in lines[1:]:
            column_indices = [int(i)-1 for i in line.split(';')[:-1][3::2]]
            row_values = [float(i) for i in line.split(';')[:-1][4::2]]
            print row_index,column_indices
            features[row_index,column_indices] = row_values
            row_index += 1
    return features

# print get_features(fe_path)
def get_csr_labels(xgmat_dataset_path):
    print ('start running example of build DMatrix from scipy.sparse CSR Matrix')
    labels = []
    row = []; col = []; dat = []
    i = 0
    for l in open(xgmat_dataset_path):
        arr = l.split()
        labels.append(int(arr[0]))
        for it in arr[1:]:
            k,v = it.split(':')
            row.append(i); col.append(int(k)); dat.append(float(v))
        i += 1
    csr = scipy.sparse.csr_matrix((dat, (row,col)))
    return csr,labels

def tune_n_estimators(alg,xgtrain,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True, show_stdv=True)
        # alg.set_params(n_estimators=cvresult.shape[0])
    return cvresult.shape[0]

if __name__ == '__main__':
    """
    fe_path = 'Data/OC-vuq2/NN_test2.txt'
    fe_path_label = 'Data/OC-vuq2/NNAI_test2.txt'
    save2xgdata(fe_path, fe_path_label)
    """
    # path_model_untuned = 'Models/OC-vuq2/untuned.xgmodel'
    path_model = 'Models/OC-vuq2/tuned_2017_04_24_14_23_23.xgmodel'
    dtest_path = 'Data/OC-vuq2/NN_test2.txt.xgmat'

    # model_untuned = xgb.Booster(model_file=path_model_untuned)
    model = xgb.Booster(model_file=path_model)

    dtest = xgb.DMatrix(dtest_path)
    labels = dtest.get_label()
    # pred_scores_un = model_untuned.predict(dtest)
    pred_scores = model.predict(dtest, output_margin=False)

    labels_list = [labels, labels]
    pred_scores_list = [pred_scores, pred_scores]
    model_names_list = ['thres_default', 'thres_new']
    thres_list = [0.5, 0.9658]
    marker_list = ['g-', 'r-']
    is_eq_greater_list = [1, 1, 1]

    measures_all, re_p_r_t_basedOnP = compare_models(labels_list, pred_scores_list, model_names_list, thres_list,
                                                     is_eq_greater_list)
    pprint.pprint(re_p_r_t_basedOnP)