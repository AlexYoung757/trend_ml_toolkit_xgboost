import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
import glob

# f1_name = './Data/features/2017-05_cmd_size.csv'
# f2_name = './Data/features/2017-05_file_size.csv'
# f3_name  = './Data/features/2017-05_ncmds.csv'
features_name = glob.glob('./Data/features/features/*csv')

train_name = './Data/features/2017-05_train.csv'
test_name = './Data/features/2017-05_test.csv'
source_name = './Data/features/2017-05_source.csv'
sample_path_name = './Data/features/2017-05_sample_path.csv'
output_train_csv = './Data/features/train.csv'
output_test_csv = './Data/features/test.csv'

output_train_svmlib = './Data/features/train.libsvm'
output_test_svmlib = './Data/features/test.libsvm'

feature_dfs = []
for i in features_name:
    print(i)
    feature_dfs.append(pd.read_csv(i))
# f1 = pd.read_csv(f1_name)
# f2 = pd.read_csv(f2_name)
# f3 = pd.read_csv(f3_name)
source_df = pd.read_csv(source_name)
train_df = pd.read_csv(train_name)
test_df = pd.read_csv(test_name)
sample_path_df = pd.read_csv(sample_path_name)

train_num = train_df.shape[0]
test_num = test_df.shape[0]

for i in feature_dfs:
    train_df = train_df.merge(right=i, how='inner', on='id', copy=False)
train_df = train_df.merge(right=source_df,how='inner',on='id',copy=False)
train_df = train_df.merge(right=sample_path_df,how='inner',on='id',copy=False)

# train_df = train_df.merge(right=f1,how='inner',on='id',copy=False)
# train_df = train_df.merge(right=f2,how='inner',on='id',copy=False)
# train_df = train_df.merge(right=f3,how='inner',on='id',copy=False)
# train_df = train_df.merge(right=source_df,how='inner',on='id',copy=False)
for i in feature_dfs:
    test_df = test_df.merge(right=i, how='inner', on='id', copy=False)
test_df = test_df.merge(right=source_df,how='inner',on='id',copy=False)
test_df = test_df.merge(right=sample_path_df,how='inner',on='id',copy=False)

# test_df = test_df.merge(right=f1,how='inner',on='id',copy=False)
# test_df = test_df.merge(right=f2,how='inner',on='id',copy=False)
# test_df = test_df.merge(right=f3,how='inner',on='id',copy=False)
# test_df = test_df.merge(right=source_df,how='inner',on='id',copy=False)

assert (train_df.shape[0] == train_num)and(test_df.shape[0] == test_num),'the shape are not compatible\n'

train_df.to_csv(path_or_buf=output_train_csv,index=False)
test_df.to_csv(path_or_buf=output_test_csv,index=False)

# train_X = train_df[['cmd_size','file_size','ncmds']].as_matrix()
train_X = train_df.drop(['id','malware','source','sample_path'],axis=1)
train_Y = train_df['malware'].as_matrix()

# test_X = test_df[['cmd_size','file_size','ncmds']].as_matrix()
test_X = test_df.drop(['id','malware','source','sample_path'],axis=1)
test_Y = test_df['malware'].as_matrix()

dump_svmlight_file(train_X,train_Y,output_train_svmlib)
dump_svmlight_file(test_X,test_Y,output_test_svmlib)




"""
train_test_df = pd.concat([train_df,test_df])

train_num = train_df.shape[0]
test_num = test_df.shape[0]

whole_df = train_test_df.merge(right=f1,how='inner',on='id',copy=False)
whole_df = whole_df.merge(right=f2,how='inner',on='id',copy=False)
whole_df = whole_df.merge(right=f3,how='inner',on='id',copy=False)
whole_df = whole_df.merge(right=source_df,how='inner',on='id',copy=False)

flag = whole_df.shape[0] == f1.shape[0] == f2.shape[0] == f3.shape[0] == source_df.shape[0]
assert flag == True,'the ids are not compatible\n'

"""









