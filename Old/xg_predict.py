import numpy as np
import xgboost as xgb
import cPickle as pickle
import argparse
import xg_plot
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file

def arg_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('-f','--feature', required=True)
    parser.add_argument('-l','--label', default='NONE')
    parser.add_argument('-x','--xgmat', default=0)
    parser.add_argument('-m','--model', required=True)
    parser.add_argument('-v', '--visual', default=30)
    return parser.parse_args()

# read label
def read_label(path):
    with open(path, 'r') as label_fi:
        label_data = []
        for line in label_fi.readlines():
            if int(line.split('|')[0]) == 2:
                label = 0
            if int(line.split('|')[0]) == 1:
                label = 1
            label_data.append(label)
        #print(label_data)
        label_data = np.asarray(label_data)

    print('Finished read label')
    return label_data

# xgboost needs data in a certain format:
# label 1:value1 2:value2 5:value5 ...
# load feature from fe_path and label from la_path, save the converted data into save_path
def save2xgdata(fe_path, la_path):
    label = read_label(la_path)
    # feature and label is in numpy.asarray format
    saver = open(fe_path+'.xgmat', 'w')
    with open(fe_path, 'r') as data_fi:
    	lines = data_fi.readlines()
    	dimension = int(lines[0].strip())
        count = 0
        for line in lines[1:]:
            data_index = [int(i) for i in line.split(';')[:-1][3::2]]
            data_value = [float(i) for i in line.split(';')[:-1][4::2]]
            data_str = str(label[count])
            count += 1
            try:
                for i in range(len(data_index)):
                    data_str += (' '+str(data_index[i])+':'+str(data_value[i]))
            except IndexError:
                pass
            if len(data_index)<dimension:
                data_str += (' '+str(dimension)+':'+str(0))
            saver.writelines(data_str+'\n')
    print('Finished read feature')

# Give preds and true labels, output performance indices including Accuracy, Precision, Recall
def evaluation(preds, label):
    preds = [int(preds[i]>=0.5) for i in range(len(preds))]
    accuracy = sum([int(preds[i]==label[i]) for i in range(len(preds))]) / (len(preds)+0.0)
    precision = sum([int(preds[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(preds)+0.0)
    recall = sum([int(label[i]==1 and preds[i]==label[i]) for i in range(len(preds))]) / (sum(label)+0.0)
    return accuracy, precision, recall

# Give test data and model, return performance indices
def xgb_predict(test, model):
    preds = model.predict(test)
    label = test.get_label()
    return preds, label


if __name__ == '__main__':
    arg = arg_parser()
    if arg.xgmat==0 and arg.label=='NONE':
        print('\nERROR, You must give label file if the data is not xgmat\n')
        exit(0)
    if int(arg.xgmat)==0:
        try:
            save2xgdata(arg.feature, arg.label)
            xg_test = arg.feature + '.xgmat'
        except:
            pass
    else:
        xg_test = arg.feature
    test = xgb.DMatrix(xg_test)
    print test.num_col()
    xg_model = pickle.load(file(arg.model, 'rb'))
    preds, label = xgb_predict(test, xg_model)
    accuracy, precision, recall = evaluation(preds, label)
    print("\nAccuracy = %.4f\nPrecision = %.4f\nRecall = %.4f\n" %(accuracy, precision, recall))
    data = load_svmlight_file(xg_test)  # for plot
    xg_plot.fea_plot(xg_model, data[0].toarray(), data[1], max_num_features=int(arg.visual))
    xg_plot.tsne_plot(xg_model, data[0].toarray(), data[1], max_num_features=int(arg.visual))
    plt.show()






