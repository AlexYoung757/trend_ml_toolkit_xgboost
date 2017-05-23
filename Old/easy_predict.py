import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score,\
    average_precision_score,classification_report,confusion_matrix,f1_score,\
    precision_score,recall_score,roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt

plt.figure(1)
plt.title('P-R curve')
plt.xlabel('Recall')
plt.ylabel('Precision')


dtest = xgb.DMatrix('Data/NN_test.txt.xgmat')
def predict_eval_model(dtest,model_path):
    labels = dtest.get_label()
    bst = xgb.Booster(model_file=model_path)
    preds_prob = bst.predict(data=dtest)
    preds_original = bst.predict(data=dtest,output_margin=True)
    preds_label = []
    for pred_prob in preds_prob:
        if pred_prob>=0.5:
            preds_label.append(1)
        else:
            preds_label.append(0)
    print 'true label:\n',labels
    from sklearn.metrics import accuracy_score
    accuracy_score = accuracy_score(labels,preds_label)
    # accuracy_score_num = accuracy_score(labels,preds_label,normalize=True)
    # print 'accuracy_score : %f\tnum of the predicted truly : %d/%d'%accuracy_score,accuracy_score_num,len(labels)
    print 'accuracy_score : %f'%accuracy_score
    print 'average_precision_score : %f'%average_precision_score(labels,preds_prob)
    print classification_report(labels,preds_label,target_names=['class 0','class 1'])
    print confusion_matrix(labels,preds_label,labels=[0,1])
    f1_score_s = f1_score(labels,preds_label,pos_label=1)
    print 'f1 score : %f'%f1_score_s
    print 'precision_score : %f'%precision_score(labels,preds_label,pos_label=1)
    print 'recall_score : %f'%recall_score(labels,preds_label,pos_label=1)
    print 'roc_auc_score  : %f'%roc_auc_score(labels,preds_prob)

    fpr, tpr, thresholds = roc_curve(labels, preds_prob)
    print fpr,tpr,thresholds
    roc_auc = auc(fpr, tpr)
    precision,recall,thresholds_pr = precision_recall_curve(labels,preds_prob,pos_label=1)
    plt.plot(precision,recall,label='P-R f1 score %f'%(f1_score_s))
    # plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

    # average_precision_score = average_precision_score(labels,preds_prob)
    # print 'average_precision_score : %f'%average_precision_score


predict_eval_model(dtest,'Models/untuned.xgmodel')
predict_eval_model(dtest,'Models/tuned.model')
""""
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
"""
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.show()