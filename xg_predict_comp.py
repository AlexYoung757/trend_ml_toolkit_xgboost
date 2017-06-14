#coding=utf-8
import xgboost as xgb
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint

from tools import get_predictedResults_from_file,read_sample_path


global num_thres
num_thres = 0
# if the score of the sample is bigger or euqal than the given threshold,then it is labeled the positive
def get_labels_by_thres(preds_prob,thres):
    labels = []
    for prob in preds_prob:
        if thres>0:
            if prob>=thres:
                labels.append(1)
            else:
                labels.append(0)
        else:
            if prob>thres:
                labels.append(1)
            else:
                labels.append(0)
    return labels

def get_labels_by_thres_eq_greater(pred_scores,thres):
    labels = []
    for score in pred_scores:
        if score >= thres:
            labels.append(1)
        else:
            labels.append(0)
    return labels
def get_labels_by_thres_greater(pred_scores,thres):
    labels = []
    for score in pred_scores:
        if score > thres:
            labels.append(1)
        else:
            labels.append(0)
    return labels
def get_precisions_recalls_tprs_fprs(labels,pred_scores,is_eq_greater):
    precisions = []
    recalls = []
    tprs = []
    fprs = []
    pred_scores_unique = list(set(pred_scores))
    total = len(pred_scores_unique)
    random_order = np.random.permutation(total).tolist()
    i = 0
    for index in random_order[:num_thres]:
        thres = pred_scores_unique[index]
        print '=========== %d / %d'%(i,num_thres)
        if is_eq_greater == 1:
            # print 'nnnnn'
            pred_labels = get_labels_by_thres_eq_greater(pred_scores,thres)
        else:
            # print 'mmmmm'
            pred_labels = get_labels_by_thres_greater(pred_scores, thres)
        con_matrix = confusion_matrix(labels,pred_labels,labels=[0,1])
        print con_matrix
        tn = con_matrix[0,0]
        fp = con_matrix[0,1]
        fn = con_matrix[1,0]
        tp = con_matrix[1,1]
        recall = float(tp)/(tp+fn)
        precision = float(tp)/(tp+fp)
        tpr = float(tp)/(tp+fn)
        fpr = float(fp)/(tn+fp)
        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)
        i=i+1
    precisions.append(1.0)
    recalls.append(0)
    precisions,recalls,tprs,fprs = sort_pr(precisions,recalls,tprs,fprs,is_p=0,is_t=0)
    return precisions,recalls,tprs,fprs
def sort_pr(precisions,recalls,tprs,fprs,is_p,is_t):
    all = []
    all_roc = []
    for i in range(len(precisions)):
        all.append((precisions[i],recalls[i]))
    for i in range(len(tprs)):
        all_roc.append((tprs[i],fprs[i]))
    if is_p:
        all.sort(key=lambda k:k[0])
    else:
        all.sort(key=lambda k:k[1])
    if is_t:
        all_roc.sort(key=lambda k:k[0])
    else:
        all_roc.sort(key=lambda k:k[1])
    p = []
    r = []
    t = []
    f = []
    for i in range(len(all)):
        p.append(all[i][0])
        r.append(all[i][1])
    for i in range(len(all_roc)):
        t.append(all_roc[i][0])
        f.append(all_roc[i][1])
    return p,r,t,f

def get_labels(preds_prob,thres,is_eq_greater=1):
    labels = []
    if is_eq_greater == 1:
        for prob in preds_prob:
            if prob >= thres:
                labels.append(1)
            else:
                labels.append(0)
    else:
        for prob in preds_prob:
            if prob > thres:
                labels.append(1)
            else:
                labels.append(0)
    return labels

def calcu_measures(labels,preds_prob,model_name,thres):

    preds_label = get_labels(preds_prob,thres)
    tn, fp, fn, tp = confusion_matrix(labels, preds_label).ravel()

    measures = dict()
    measures['model_name'] = model_name
    # 这些值都是基于一定的切分阈值计算出来的
    measures['tn'] = tn
    measures['fp'] = fp
    measures['fn'] = fn
    measures['tp'] = tp
    measures['accuracy'] = accuracy_score(labels,preds_label)
    measures['classification_report'] = classification_report(labels,preds_label,target_names=['0','1'])
    measures['confusion_matrix'] = confusion_matrix(labels,preds_label,labels=[0,1])
    measures['f1'] = f1_score(labels,preds_label,pos_label=1)
    measures['precision'] = precision_score(labels,preds_label,pos_label=1)
    measures['recall'] = recall_score(labels,preds_label,pos_label=1)
    # 这些无需基于阈值
    measures['fprs'], measures['tprs'], measures['thresholds_roc'] = roc_curve(labels, preds_prob,pos_label=1)
    measures['precisions'], measures['recalls'], measures['thresholds_pr'] = precision_recall_curve(labels,preds_prob,pos_label=1)
    measures['roc_auc'] = auc(measures['fprs'], measures['tprs'])
    measures['pr_auc'] = auc(measures['recalls'], measures['precisions'])# 对于pr_auc的计算，上下两种方式都可以
    # measures['pr_auc'] = average_precision_score(labels, preds_prob)

    return measures
"""
def combine_measures(measures0,measures1):
    measures = dict()
    for k,v in measures0.iteritems():
        v1 = measures1[k]
        measures[k] = (v,v1)
    return measures
"""
def combine_measuresList(measures_list):
    measures = dict()
    num_measures_list = len(measures_list)
    for k,v in measures_list[0].iteritems():
        measures[k] = []
        for i in range(num_measures_list):
            measures[k].append(measures_list[i][k])
    return measures

"""
def compare_roc_auc(measures):
    plt.figure(1)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0,1,0,1])
    # plt.plot(measures['fpr'][0],measures['tpr'][0],'r-',measures['fpr'][0],measures['tpr'][0],'ro',label='%s roc_auc : %f'%(measures['model_name'][0], measures['roc_auc'][0]))
    plt.plot(measures['fpr'][0],measures['tpr'][0],'r-',label='%s roc_auc : %f'%(measures['model_name'][0], measures['roc_auc'][0]))
    # plt.plot(measures['fpr'][1], measures['tpr'][1], 'g-', measures['fpr'][1], measures['tpr'][1], 'go',label='%s roc_auc : %f' % (measures['model_name'][1], measures['roc_auc'][1]))
    plt.plot(measures['fpr'][1], measures['tpr'][1], 'g-',label='%s roc_auc : %f' % (measures['model_name'][1], measures['roc_auc'][1]))
    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.show()
"""
def compare_roc_auc(measures,fid=1,axis_interval=[0,1,0,1],marker_list=['g-','r-','b-']):
    plt.figure(fid)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis(axis_interval)
    for i in range(len(measures['fprs'])):
        # plt.plot(measures['fpr'][0],measures['tpr'][0],'r-',measures['fpr'][0],measures['tpr'][0],'ro',label='%s roc_auc : %f'%(measures['model_name'][0], measures['roc_auc'][0]))
        # plt.plot(measures['fprs'][i],measures['tprs'][i],marker_list[i],label='%s roc_auc : %f'%(measures['model_name'][i], measures['roc_auc'][i]))
        plt.plot(measures['fprs'][i],measures['tprs'][i],marker_list[i],label='%s roc_auc '%(measures['model_name'][i]))
        # plt.plot(measures['fpr'][1], measures['tpr'][1], 'g-', measures['fpr'][1], measures['tpr'][1], 'go',label='%s roc_auc : %f' % (measures['model_name'][1], measures['roc_auc'][1]))
    plt.legend(loc='lower right', shadow=True)
    plt.show()
def compare_pr_auc(measures,fid=1,axis_interval=[0,1,0,1],marker_list=['g-','r-','b-']):
    plt.figure(fid)
    plt.title('P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis(axis_interval)
    for i in range(len(measures['precisions'])):
        plt.plot(measures['recalls'][i],measures['precisions'][i],marker_list[i],label='%s pr_auc '%(measures['model_name'][i]))
        # plt.plot(measures['precisions'][i],measures['recalls'][i],marker_list[i],label='%s pr_auc : %f'%(measures['model_name'][i], measures['pr_auc'][i]))
        # plt.plot(measures['precisions'][0],measures['recalls'][0],'r-',measures['precisions'][0],measures['recalls'][0],'ro',label='%s pr_auc : %f'%(measures['model_name'][0], measures['pr_auc'][0]))
        # plt.plot(measures['precisions'][1], measures['recalls'][1], 'g-', label='%s pr_auc : %f' % (measures['model_name'][1], measures['pr_auc'][1]))
        # plt.plot(measures['precisions'][1], measures['recalls'][1], 'g-', measures['precisions'][1], measures['recalls'][1], 'go',label='%s pr_auc : %f' % (measures['model_name'][1], measures['pr_auc'][1]))
    # plt.legend(loc='lower center', shadow=True, fontsize='x-large')
    plt.legend(loc='lower center', shadow=True)

    plt.show()
def compare_confusion_matrix(measures):
    # con_mat = {'tp':measures['tp'],'tn':measures['tn'],'fp':measures['fp'],'fn':measures['fn'],'model_name':measures['model_name']}
    con_mat = {'tp':measures['tp'],'tn':measures['tn'],'fp':measures['fp'],'fn':measures['fn']}
    indice = [name for name in measures['model_name']]
    df = pd.DataFrame(data=con_mat,index=indice)
    print df
"""
def compare_model(labels,preds_prob0,preds_prob1,model_name0,model_name1):
    measures0 = calcu_measures(labels=labels,preds_prob=preds_prob0,model_name=model_name0)
    measures1 = calcu_measures(labels=labels,preds_prob=preds_prob1,model_name=model_name1)
    measures = combine_measures(measures0,measures1)
    df = pd.DataFrame(measures,index=[measures['model_name'][0],measures['model_name'][1]])
    df = df.drop(['model_name','classification_report','confusion_matrix','fpr','precisions','recalls','thresholds_pr','thresholds_roc','tpr'],axis=1)
    print df.T
    # compare_roc_auc(measures)
    # compare_pr_auc(measures)
"""
def compare_models(labels_list,pred_scores_list,model_names_list,thres_list):
    measures_list = []
    # 度量计算
    for i in range(len(pred_scores_list)):
        measures = calcu_measures(labels=labels_list[i],preds_prob=pred_scores_list[i],model_name=model_names_list[i],thres=thres_list[i])
        measures_list.append(measures)
    measures_all = combine_measuresList(measures_list)
    df = pd.DataFrame(measures_all,index=[measures_all['model_name'][i] for i in range(len(pred_scores_list))])
    df = df.drop(['model_name','classification_report','confusion_matrix','fprs','tprs','precisions','recalls','thresholds_pr','thresholds_roc'],axis=1)
    # df = df.drop(['model_name','classification_report','confusion_matrix','precisions','recalls','fprs','tprs'],axis=1)
    print df.T
    return measures_all
def get_score(path,score):
    path = np.reshape((-1,1))
    score = np.reshape((-1,1))
    whole = np.concatenate((path,score),axis=1)
    return whole

def compare_path_score(label_path_list,scores_list,model_name_list):
    assert len(label_path_list) == len(scores_list),'数目不等！'
    all = []
    for i in range(len(label_path_list)):
        case_label_path = label_path_list[i]
        if(case_label_path != '_'):
            this_model = get_score(read_sample_path(case_label_path),scores_list[i])
            df = pd.DataFrame(this_model)
            print('======  %s  ===='%(model_name_list[i]))
            print(df)
            all.append(this_model)
    return all

def IsListSorted_sorted(lst):
    return sorted(lst) == lst or sorted(lst, reverse=True) == lst
if __name__ == '__main__':
    path_model_untuned = 'Models/untuned.xgmodel'
    path_model = 'Models/tuned.xgmodel'
    dtest_path = 'Data/NN_test.txt.xgmat'
    model_untuned = xgb.Booster(model_file=path_model_untuned)
    model = xgb.Booster(model_file=path_model)
    dtest = xgb.DMatrix(dtest_path)
    labels = dtest.get_label()
    pred_scores_un = model_untuned.predict(dtest)
    pred_scores = model.predict(dtest)
    # compare_model(labels,pred_scores_un,pred_scores,'untuned','tuned')
    svm_labels, pred_scores_svm = get_predictedResults_from_file('Data/rslt_app.txt')
    labels_list = [labels,labels,svm_labels]
    pred_scores_list = [pred_scores_un,pred_scores,pred_scores_svm]
    model_names_list = ['untuned_xgb','tuned_xgb','svm']
    thres_list = [0.5,0.5,0.0]
    marker_list = ['g-','r-','b-']
    is_eq_greater_list = [1,1,1]
    measures_all = compare_models(labels_list,pred_scores_list,model_names_list,thres_list,is_eq_greater_list)
    temp = measures_all['recalls'][0]
    # print temp
    # print IsListSorted_sorted(temp)


    # print IsListSorted_sorted(measures_all['precisions'][0])
    # print measures_all['recalls'][2]
    # print measures_all['precisions'][2]
    compare_roc_auc(fid=1,measures=measures_all,marker_list=marker_list)
    compare_pr_auc(fid=2, measures=measures_all, marker_list=marker_list)
    compare_roc_auc(fid=3, measures=measures_all,axis_interval=[0,1,0.9,1],marker_list=marker_list)
    compare_pr_auc(fid=4, measures=measures_all, axis_interval=[0,1,0.9,1],marker_list=marker_list)