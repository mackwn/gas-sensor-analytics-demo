import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
from scipy import stats

def split_data(input_fn,train_fn='train_data', test_fn='test_data', cv_fn='cv_data', test_size=.4):
    df = pd.read_pickle('data/interim/{}.pkl'.format(input_fn))
    # Remove outliers - see initial exploration notebook for setting Z filter
    # Unimplemented for now since it does not appear to be helping the model
    ##df = df[(np.abs(stats.zscore(df.loc[:,df.columns.values[3:]])) < 2).all(axis=1)].copy()
    train, testing = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    cross_validation, test = train_test_split(testing, test_size=0.5, random_state=42, shuffle=True)

    train.to_pickle('data/processed/{}.pkl'.format(train_fn))
    test.to_pickle('data/processed/{}.pkl'.format(test_fn))
    cross_validation.to_pickle('data/processed/{}.pkl'.format(cv_fn))

if __name__ == "__main__":
    split_data('merged_data','train_data','test_data','cv_data')