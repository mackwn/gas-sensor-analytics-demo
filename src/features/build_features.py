import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class filterUnusualX(BaseEstimator, TransformerMixin):
    def __init__(self, z_score_max=4):
        self.z_score_max = z_score_max
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        X_ = np.where(abs(X) > self.z_score_max, np.nan, X)
        return X_

# Pipeline to prepare feature array for regression
def preprocess_pipeline_reg():

    cat_idx = [129]
    num_idx = list(range(0,129))

    trans_encode_scale = ColumnTransformer(
        [
            ('cat', OneHotEncoder(), cat_idx), ('num', PowerTransformer(), num_idx)
        ]
    )

    trans_filter = ColumnTransformer([
        ('filter', filterUnusualX(), list(range(0,129)))
        ], remainder='passthrough'
    )

    # trans_impute = ColumnTransformer(
    #     [
    #         ('impute', KNNImputer(), list(range(0,129)))
    #     ]
    #     , remainder='passthrough'
    # )

    pl = Pipeline(steps=[
        ('scale_encode', trans_encode_scale),
        ('filter', trans_filter),
        ('impute', KNNImputer())
    ])

    return pl

# Pipeline to prepare feature array data for
def preprocess_pipeline_clf():
    pl = Pipeline(steps=[
        ('scale', PowerTransformer()),
        ('filter', filterUnusualX(4)),
        ('impute', KNNImputer())
    ])

    return pl

def df_to_arr_clf(df):
    data = df.values
    X = data[:,3:]
    y = data[:,1] 
    return X, y 

def df_to_arr_reg(df):
    data = df.values
    X = np.concatenate((data[:,3:],  data[:,1:2]), axis=1)
    y = data[:,2] 
    return X, y 


def encode_categorical(data, output_fn):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(data)

    pickle.dump(encoder,open('models/preprocessing/{}.pkl'.format(output_fn),'wb'))


def scale_data(data, output_fn):
    #scaler = StandardScaler()
    
    scaler = PowerTransformer()
    scaler.fit(data)
    X_scaled = scaler.transform(X)

    pickle.dump(scaler,open('models/preprocessing/{}.pkl'.format(output_fn),'wb'))

def reduce_dimensions(data, scaler_fn, output_fn):
    # 56 components selected for 99.9% explained variance - see notebook
    scaler = pickle.load(open('models/preprocessing/{}.pkl'.format(scaler_fn),'rb'))
    X_scaled = scaler.transform(data)
    pca = PCA(56)
    pca.fit(X_scaled)

    pickle.dump(pca,open('models/preprocessing/{}.pkl'.format(output_fn),'wb'))

if __name__ == "__main__":
    df = pd.read_pickle('data/processed/train_data.pkl')
    data = df.values
    X_clf = data[:,3:]
    X_reg = np.concatenate((X_clf,data[:,1:2]), axis=1)

    pl_reg = preprocess_pipeline_reg()
    pl_clf = preprocess_pipeline_clf()

    pl_reg.fit_transform(X_reg)
    pl_clf.fit_transform(X_clf)


    #scale_data(X,'scaler')
    #reduce_dimensions(X,'scaler','pca')
    #encode_categorical(X_cat, 'encoder')
