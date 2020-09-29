import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import src.features.build_features as build_features


def train_classification(df, classifier, out_model_fn):

    X, y = build_features.df_to_arr_clf(df)
    pl = build_features.preprocess_pipeline_clf()
    pl.steps.append(('model', classifier))
    pl.fit(X, y)
    pickle.dump(pl,
        open('models/classification/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )
    return pl

def train_regression(df, regressor, out_model_fn):

    X, y = build_features.df_to_arr_reg(df)
    pl = build_features.preprocess_pipeline_reg()
    pl.steps.append(('model', regressor))
    pl.fit(X, y)
    pickle.dump(pl,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )
    return pl

if __name__ == "__main__":
    df_train = pd.read_pickle('data/processed/train_data.pkl')
    train_classification(df_train, SVC(C=500), 'clf_svc')
    train_regression(df_train, SVR(C=2000), 'clf_svr')