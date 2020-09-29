import sys
sys.path.append(".")

import src.features.build_features as build_features
import numpy as np


def predict_gas(X, pipeline_fn):

    pl = pickle.load(
        open('models/classification/{}.pkl'.format(pipeline_fn),'rb')
    )
    y_pred = pl.predict(X, y)
    return y_pred


def predict_conc(X, pipeline_fn):

    pl = pickle.load(
        open('models/regression/{}.pkl'.format(pipeline_fn),'rb')
    )
    y_pred = pl.predict(X)
    return y_pred

def predict_gas_conc(X, pipeline_fn_clf, pipeline_fn_reg):

    pl_clf = pickle.load(
        open('models/classification/{}.pkl'.format(pipeline_fn),'rb')
    )
    y_gas_pred = pl_clf.predict(X)

    X = np.concatenate((X, y_gas_pred), axis=1)
    pl_reg = pickle.load(
        open('models/regression/{}.pkl'.format(pipeline_fn),'rb')
    )
    y_conc_pred = pl_reg.predict(X)

    return y_pred

if __name__ == "__main__":
    
    pass