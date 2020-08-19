import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_data_(data_fn, scaler_fn):
    df = pd.read_pickle('data/processed/{fn}.pkl'.format(fn=data_fn))
    data = df.values
    batch_id = data[:,0]
    y = data[:,1]
    X = data[:,3:]

    scaler = pickle.load(
    open('models/preprocessing/{fn}.pkl'.format(fn=scaler_fn),'rb')
    )

    X = scaler.transform(X)

    return X,y

def train_logistic(data_fn, scaler_fn, out_model_fn, C=125):
    X,y = load_data_(data_fn, scaler_fn)
    clf = LogisticRegression(random_state=0, max_iter=5000, C=C).fit(X, y)
    pickle.dump(clf,
        open('models/classification/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )
    
def train_svmc(data_fn, scaler_fn, out_model_fn, C=400):
    X,y = load_data_(data_fn, scaler_fn)
    clf = SVC(C=C, max_iter=5000)
    clf.fit(X,y)
    pickle.dump(clf,
        open('models/classification/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )

def train_randomforest(data_fn, scaler_fn, out_model_fn, max_depth=12):
    X,y = load_data_(data_fn, scaler_fn)
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
    pickle.dump(clf,
        open('models/classification/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )

if __name__ == "__main__":
    scaler_fn = 'scaler'
    data_fn = 'train_data'
    train_logistic(data_fn,scaler_fn,'logist_reg')
    train_randomforest(data_fn,scaler_fn,'rf_class')
    train_svmc(data_fn,scaler_fn,'svm_class')