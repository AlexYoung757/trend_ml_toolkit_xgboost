import argparse
import ConfigParser
from tools import save2xgdata
import xgboost as xgb
# parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--conf', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()

# configure parser
def conf_parser(conf_path):
    cf = ConfigParser.ConfigParser()
    cf.read(conf_path)

    booster = cf.get('xg_train_untuned_conf', 'booster')
    silent = cf.getint('xg_train_untuned_conf', 'silent')
    nthread = cf.getint('xg_train_untuned_conf', 'nthread')

    objective = cf.get('xg_train_untuned_conf', 'objective')
    dtrain_path = cf.get('xg_train_untuned_conf', 'dtrain')
    if cf.getint('xg_train_untuned_conf','xgmat')==0: # if it is not a xgmat file, than convert it
        try:
            label = cf.get('xg_train_untuned_conf', 'label')
            data = cf.get('xg_train_untuned_conf','data')
            save2xgdata(data, label)
            dtrain_path = data+'.xgmat'
        except:
            pass
    params = {'booster': booster, 'objective': objective, 'silent': silent,'nthread': nthread}
    return params, dtrain_path

if __name__ == '__main__':
    args = arg_parser()
    params, dtrain_path = conf_parser(args.conf)
    print params,dtrain_path,args
    dtrain = xgb.DMatrix(dtrain_path)
    xg_model = xgb.train(params,dtrain)
    xg_model.save_model(args.output)