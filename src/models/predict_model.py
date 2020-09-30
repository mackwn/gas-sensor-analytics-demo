import sys
sys.path.append(".")
import pickle
import src.features.build_features as build_features
import numpy as np
import pandas as pd


def load_pipeline_clf(pipeline_fn):
    pl = pickle.load(
        open('models/classification/{}.pkl'.format(pipeline_fn),'rb')
    )
    return pl

def load_pipeline_reg(pipeline_fn):
    pl = pickle.load(
        open('models/regression/{}.pkl'.format(pipeline_fn),'rb')
    )
    return pl


def predict_gas_conc(X, pl_clf, pl_reg):

    y_gas_pred = pl_clf.predict(X)
    y_gas_pred = y_gas_pred.reshape(y_gas_pred.shape[0],1)
    X = np.concatenate((X, y_gas_pred), axis=1)
    y_conc_pred = pl_reg.predict(X)
    y_conc_pred = y_conc_pred.reshape(y_conc_pred.shape[0],1)

    return np.concatenate((y_gas_pred, y_conc_pred), axis=1)

if __name__ == "__main__":
    
    pl_clf = load_pipeline_clf('clf_svc')
    pl_reg = load_pipeline_reg('reg_svr')
    df_test = pd.read_pickle('data/processed/test_data.pkl')
    X_test, y_test = build_features.df_to_arr_clf(df_test)
    y_test_pred = predict_gas_conc(X_test, pl_clf, pl_reg)