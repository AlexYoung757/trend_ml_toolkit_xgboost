import numpy as np
import random
import xgboost as xgb
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from operator import itemgetter

def get_fea_index(fscore, max_num_features):
    fea_index = list()
    for key in fscore:
        fea_index.append(int(key[0][1:])-1) # note that the index of array in Python start from 0, while xg_feature start from f1
    if max_num_features==None:
        pass
    else:
        fea_index = fea_index[0:max_num_features]
    return np.array(fea_index)

def fea_plot(xg_model, feature, label, type = 'weight', max_num_features = None):
    fig, AX = plt.subplots(nrows=1, ncols=2)
    xgb.plot_importance(xg_model, xlabel=type, importance_type='weight', ax=AX[0], max_num_features=max_num_features)

    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True) # sort scores
    fea_index = get_fea_index(fscore, max_num_features)
    feature = feature[:, fea_index]
    dimension = len(fea_index)
    X = range(1, dimension+1)
    Yp = np.mean(feature[np.where(label==1)[0]], axis=0)
    Yn = np.mean(feature[np.where(label!=1)[0]], axis=0)
    for i in range(0, dimension):
        param = np.fmax(Yp[i], Yn[i])
        Yp[i] /= param
        Yn[i] /= param
    p1 = AX[1].bar(X, +Yp, facecolor='#ff9999', edgecolor='white')
    p2 = AX[1].bar(X, -Yn, facecolor='#9999ff', edgecolor='white')
    AX[1].legend((p1,p2), ('Malware', 'Normal'))
    AX[1].set_title('Comparison of selected features by their means')
    AX[1].set_xlabel('Feature Index')
    AX[1].set_ylabel('Mean Value')
    AX[1].set_ylim(-1.1, 1.1)
    plt.xticks(X, fea_index+1, rotation=80)
    plt.suptitle('Feature Selection results')

def setcolor(index):
    color = np.zeros([index.shape[0], 4])
    for i in range(0, index.shape[0]):
        if index[i] == 1:
            color[i, 0] = 1
        else:
            color[i, 2] = 1
        color[i, 3] = 1
    return color

def tsne_plot(xg_model, feature, label, type = 'weight', max_num_features = None, size=200):
    fig, AX = plt.subplots(nrows=1, ncols=2)
    fscore = xg_model.get_score(importance_type=type)
    fscore = sorted(fscore.items(), key=itemgetter(1), reverse=True)  # sort scores
    fea_index = get_fea_index(fscore, max_num_features)
    # select points for plotting in size:size
    positive = feature[np.where(label==1)[0]]; pos_label = label[np.where(label==1)[0]]
    negative = feature[np.where(label!=1)[0]]; neg_label = label[np.where(label!=1)[0]]

    # In positive
    num_pos = positive.shape[0]
    step = np.ceil(num_pos / size)
    pos_index = list()
    for i in range(1, size+1):
        if i*step<=num_pos:
            pos_index.append((i-1)*step+int(random.uniform(0, step-0.0000001)))
        else:
            pos_index.append(int(random.uniform((i-1)*step, num_pos - 0.0000001)))
    pos_index = np.asarray(pos_index, np.int8)
    positive = positive[pos_index,:]
    pos_label = pos_label[pos_index]

    # In negative
    num_neg = negative.shape[0]
    step = np.ceil(num_neg / size)
    neg_index = list()
    for i in range(1, size+1):
        if i*step<=num_neg:
            neg_index.append((i-1)*step+int(random.uniform(0, step-0.0000001)))
        else:
            neg_index.append(int(random.uniform((i-1)*step, num_neg - 0.0000001)))
    neg_index = np.asarray(neg_index, np.int8)
    negative = negative[neg_index,:]
    neg_label = neg_label[neg_index]

    feature = np.concatenate([positive, negative])
    label = np.concatenate([pos_label, neg_label])

    tsne_origin = TSNE(learning_rate=100).fit_transform(feature, label)
    tsne_trans = TSNE(learning_rate=100).fit_transform(feature[:, fea_index], label)

    p1 = AX[0].scatter(tsne_origin[:size, 0], tsne_origin[:size, 1], c=setcolor(label[:size]))
    p2 = AX[0].scatter(tsne_origin[size:, 0], tsne_origin[size:, 1], c=setcolor(label[size:]))
    AX[0].legend((p1, p2), ('Malware', 'Normal'), scatterpoints=1)
    AX[0].set_title('Low dimensional structure of original feature space')
    p3 = AX[1].scatter(tsne_trans[:size, 0], tsne_trans[:size, 1], c=setcolor(label[:size]))
    p4 = AX[1].scatter(tsne_trans[size:, 0], tsne_trans[size:, 1], c=setcolor(label[size:]))
    AX[1].legend((p3, p4), ('Malware', 'Normal'), scatterpoints=1)
    AX[1].set_title('Low dimensional structure of selected feature space\nSelected by '+type)
    plt.suptitle('Visualized feature space')
