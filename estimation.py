from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt


def get_labels_by_thres(preds_prob,thres):
    labels = []
    for prob in preds_prob:
        if prob>=0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def calcu_measures(labels,preds_prob,model_name):
    preds_label = get_labels_by_thres(preds_prob,0.5)
    measures = dict()
    measures['model_name'] = model_name
    measures['accuracy'] = accuracy_score(labels,preds_label)
    measures['pr_auc'] = average_precision_score(labels,preds_prob)
    measures['classification_report'] = classification_report(labels,preds_label,target_names=['0','1'])
    measures['confusion_matrix'] = confusion_matrix(labels,preds_label,labels=[0,1])
    measures['f1'] = f1_score(labels,preds_label,pos_label=1)
    measures['precision'] = precision_score(labels,preds_label,pos_label=1)
    measures['recall'] = recall_score(labels,preds_label,pos_label=1)
    measures['fpr'], measures['tpr'], measures['thresholds_roc'] = roc_curve(labels, preds_prob)
    measures['precisions'], measures['recalls'], measures['thresholds_pr'] = precision_recall_curve(labels,preds_prob,pos_label=1)
    measures['roc_auc'] = auc(measures['fpr'], measures['tpr'])
    return measures
def combine_measures(measures0,measures1):
    measures = dict()
    for k,v in measures0.iteritems():
        v1 = measures1[k]
        measures[k] = (v,v1)
    return measures
def compare_roc_auc(measures):
    plt.figure(1)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0,1,0,1])
    plt.plot(measures['fpr'][0],measures['tpr'][0],'r-',measures['fpr'][0],measures['tpr'][0],'ro',label='%s roc_auc : %f'%(measures['model_name'][0], measures['roc_auc'][0]))
    plt.plot(measures['fpr'][1], measures['tpr'][1], 'g-', measures['fpr'][1], measures['tpr'][1], 'go',label='%s roc_auc : %f' % (measures['model_name'][1], measures['roc_auc'][1]))
    plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.show()

def compare_pr_auc(measures):
    plt.figure(2)
    plt.title('P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0,1,0,1])
    plt.plot(measures['precisions'][0],measures['recalls'][0],'r-',measures['precisions'][0],measures['recalls'][0],'ro',label='%s pr_auc : %f'%(measures['model_name'][0], measures['pr_auc'][0]))
    plt.plot(measures['precisions'][1], measures['recalls'][1], 'g-', measures['precisions'][1], measures['recalls'][1], 'go',label='%s pr_auc : %f' % (measures['model_name'][1], measures['pr_auc'][1]))
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.show()

def compare_model(labels,preds_prob0,preds_prob1,model_name0,model_name1):
    measures0 = calcu_measures(labels=labels,preds_prob=preds_prob0,model_name=model_name0)
    measures1 = calcu_measures(labels=labels,preds_prob=preds_prob1,model_name=model_name1)
    measures = combine_measures(measures0,measures1)
    df = pd.DataFrame(measures,index=[measures['model_name'][0],measures['model_name'][1]])
    df = df.drop(['model_name','classification_report','confusion_matrix','fpr','precisions','recalls','thresholds_pr','thresholds_roc','tpr'],axis=1)
    print df.T
    compare_roc_auc(measures)
    compare_pr_auc(measures)