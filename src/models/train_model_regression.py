import pandas as pd
import numpy as np
import os
import pickle
from sklearn.svm import SVR

def load_data_(data_fn, scaler_fn):
    df = pd.read_pickle('data/processed/{fn}.pkl'.format(fn=data_fn))
    data = df.values
    batch_id = data[:,0]
    y = data[:,2]
    X = data[:,3:]
    X_cat = data[:,1:2]

    scaler = pickle.load(
        open('models/preprocessing/{fn}.pkl'.format(fn=scaler_fn),'rb')
    )

    X = scaler.transform(X)

    return X, X_cat, y

def encode_categorical_(encoder_fn, X, X_cat):
    encoder = pickle.load(
        open('models/preprocessing/{fn}.pkl'.format(fn=encoder_fn),'rb')
    )
    X_cat = encoder.transform(X_cat)

    return np.concatenate((X,X_cat),axis=1)

def train_svmr(data_fn, scaler_fn, encoder_fn, out_model_fn, C=5000):
    X, X_cat, y = load_data_(data_fn, scaler_fn)
    X = encode_categorical_(encoder_fn, X, X_cat)
    reg_svr = SVR(C=C)
    reg_svr.fit(X,y)
    pickle.dump(reg_svr,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )

if __name__ == "__main__":
    scaler_fn = 'scaler'
    data_fn = 'train_data'
    encoder_fn = 'encoder'
    train_svmr(data_fn,scaler_fn,encoder_fn,'svm_reg')