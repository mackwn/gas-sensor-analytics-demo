import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
#from src.helpers import df_to_array

def encode_categorical(data, output_fn):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(data)

    pickle.dump(encoder,open('models/preprocessing/{}.pkl'.format(output_fn),'wb'))


def scale_data(data, output_fn):
    scaler = StandardScaler()
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
    X = data[:,3:]
    X_cat = data[:,1:2]
    scale_data(X,'scaler')
    reduce_dimensions(X,'scaler','pca')
    encode_categorical(X_cat, 'encoder')
