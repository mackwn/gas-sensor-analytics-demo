import pandas as pd
import numpy as np
import os
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

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

def train_gbr(data_fn, scaler_fn, encoder_fn, out_model_fn):
    X, X_cat, y = load_data_(data_fn, scaler_fn)
    X = encode_categorical_(encoder_fn, X, X_cat)
    reg = GradientBoostingRegressor(loss='lad',max_depth=50,n_estimators=125,learning_rate=.15)
    reg.fit(X,y)
    pickle.dump(reg,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )

def train_nn(data_fn, scaler_fn, encoder_fn, out_model_fn):
    X, X_cat, y = load_data_(data_fn, scaler_fn)
    X = encode_categorical_(encoder_fn, X, X_cat)
    reg =MLPRegressor(
        max_iter=1000,hidden_layer_sizes=tuple(4*[136]),alpha=.008,
        learning_rate_init=.007, random_state=10
    )
    reg.fit(X,y)
    pickle.dump(reg,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )

def train_rf(data_fn, scaler_fn, encoder_fn, out_model_fn):
    X, X_cat, y = load_data_(data_fn, scaler_fn)
    X = encode_categorical_(encoder_fn, X, X_cat)
    reg = RandomForestRegressor(n_estimators=200, random_state=10, max_features ='log2')
    reg.fit(X,y)
    pickle.dump(reg,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )



if __name__ == "__main__":
    scaler_fn = 'scaler'
    data_fn = 'train_data'
    encoder_fn = 'encoder'
    train_gbr(data_fn,scaler_fn,encoder_fn,'gbr_reg')
    train_nn(data_fn,scaler_fn,encoder_fn,'nn_reg')
    train_rf(data_fn,scaler_fn,encoder_fn,'rf_reg')