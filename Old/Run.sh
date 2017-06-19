# Train
# python xg_train.py -c xg_train.conf -o XG.xgmodel -v 30

# Predict
# python xg_predict.py -f Data/NN_test.txt -l Data/NNAI_test.txt -x 0 -m XG.xgmodel -v 30

# CV
# python xg_kfold.py -c xg_kfold.conf -k 10 -s 1

# train untuned
python xg_train_untuned.py -c xg_train_untuned.config -o Models/temp.xgmodel

# train tuned
python xg_train.py -c xg_train.config -o Models/temp.xgmodel

# convert data format
python tools.py -s ./Data/OC-vuq1/NN_test.txt -l ./Data/OC-vuq1/NNAI_test.txt

# compare model
python compare.py  -c compare.config

