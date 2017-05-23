import xgboost as xgb
import argparse
import ConfigParser
import pprint
import pickle
from xgboost import XGBClassifier

from tools import get_predictedResults_from_file,get_csr_labels
from xg_predict_comp import compare_models,compare_roc_auc,compare_pr_auc,compare_confusion_matrix

# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=False)
    # parser.add_argument('-v', '--visual', default=30)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path,is_verbose=False):
    global models_path
    global model_names_list
    global thres_list
    global marker_list
    global dataset_list
    global dataset_f_list

    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    models_path = cf.get('compare','model_paths').split(',')
    model_names_list = cf.get('compare', 'model_names').split(',')
    thres_list = cf.get('compare','thres').split(',')
    thres_list = [float(i) for i in thres_list]
    marker_list = cf.get('compare','markers').split(',')
    dataset_list = cf.get('compare','datasets').split(',')
    dataset_f_list = cf.get('compare','dataset_formats').split(',')

    if is_verbose:
        pprint.pprint(models_path)
        pprint.pprint(model_names_list)
        pprint.pprint(marker_list)
    return

if __name__ == '__main__':

    models_path = None
    model_names_list = None
    thres_list = None
    marker_list = None
    dataset_list = None
    dataset_f_list = None

    arg = arg_parser()
    conf_parser(arg.conf,is_verbose=True)

    # path_model_untuned = 'Models/OC-vuq2/untuned.xgmodel'
    # path_model = 'Models/OC-vuq2/tuned_2017_04_25_14_10_31.xgmodel'
    # dtest_path = 'Data/OC-vuq2/NN_train.txt.xgmat'

    labels_list = list()
    pred_scores_list = list()

    for i,f in enumerate(dataset_f_list):
        print('=============================')
        if f  == 'xgboost':
            print(dataset_list[i])
            print(models_path[i])
            dtest = xgb.DMatrix(dataset_list[i])
            labels = dtest.get_label()
            labels = labels.astype(int)
            model_temp = xgb.Booster(model_file=models_path[i])
            labels_pred = model_temp.predict(dtest)
            labels_list.append(labels)
            pred_scores_list.append(labels_pred)
        elif f == 'xgb_pickle':
            print(dataset_list[i])
            print(models_path[i])
            x,y = get_csr_labels(dataset_list[i])
            model_temp = pickle.load(open(models_path[i],'r'))
            labels_pred = model_temp.predict(x)
            labels_list.append(y)
            pred_scores_list.append(labels_pred)
        elif f in 'svm':
            print(dataset_list[i])
            print(models_path[i])
            svm_labels, pred_scores_svm = get_predictedResults_from_file(dataset_list[i])
            labels_list.append(svm_labels)
            pred_scores_list.append(pred_scores_svm)

    # model_untuned = xgb.Booster(model_file=path_model_untuned)
    # model = xgb.Booster(model_file=path_model)

    # dtest = xgb.DMatrix(dtest_path)
    # labels = dtest.get_label()
    # pred_scores_un = model_untuned.predict(dtest)
    # pred_scores = model.predict(dtest,output_margin=False)
    # pred_scores_t = model.predict(dtest,output_margin=False)
    # re = zip(pred_scores.tolist(),pred_scores_t.tolist())
    # pprint.pprint(pred_scores)
    # pprint.pprint(pred_scores_un)
    # print("*****  pred_scores    *******")
    # pprint.pprint(re)
    # svm_labels, pred_scores_svm = get_predictedResults_from_file('Data/OC-vuq2/rslt_app2.txt')

    # labels_list = [labels,labels,svm_labels]
    # pred_scores_list = [pred_scores_un,pred_scores,pred_scores_svm]
    # print len(pred_scores_un),len(pred_scores),len(pred_scores_svm)
    # print pred_scores_un[:10],pred_scores[:10],pred_scores_svm[:10]
    # model_names_list = ['untuned_xgb','tuned_xgb','svm']
    # thres_list = [0.5,0.5,0.0]
    # marker_list = ['g-','r-','b-']
    # is_eq_greater_list = [1,1,1]
    print(labels_list,pred_scores_list)
    measures_all = compare_models(labels_list,pred_scores_list,model_names_list,thres_list)
    compare_roc_auc(fid=1,measures=measures_all,marker_list=marker_list)
    compare_pr_auc(fid=2, measures=measures_all, marker_list=marker_list)
    compare_roc_auc(fid=3, measures=measures_all,axis_interval=[0,1,0.9,1],marker_list=marker_list)
    compare_pr_auc(fid=4, measures=measures_all, axis_interval=[0,1,0.9,1],marker_list=marker_list)
    compare_confusion_matrix(measures_all)