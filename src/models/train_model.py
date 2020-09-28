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



def train_classification(X, y, classifier, out_model_fn):
    pl = build_features.preprocess_pipeline_clf()
    pl.steps.append(('model', classifier))
    pl.fit(X)
    pickle.dump(pl,
        open('models/classification/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )
    return pl

def train_regression(X, y, regressor, out_model_fn)
    pl = build_features.preprocess_pipeline_reg()
    pl.steps.append(('model', regressor))
    pl.fit(X)
    pickle.dump(pl,
        open('models/regression/{fn}.pkl'.format(fn=out_model_fn),'wb')
    )
    return pl

