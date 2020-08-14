import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
#from src.helpers import df_to_array


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
    scale_data(X,'scaler')
    reduce_dimensions(X,'scaler','pca')
