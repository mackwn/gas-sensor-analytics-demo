import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
from scipy import stats

def split_data(input_fn,train_fn, test_fn, cv_fn, test_size=.4):
    df = pd.read_pickle('data/interim/{}'.format(input_fn))
    # Remove outliers - see initial exploration notebook for setting Z filter
    # Unimplemented for now since it does not appear to be helping the model
    ##df = df[(np.abs(stats.zscore(df.loc[:,df.columns.values[3:]])) < 2).all(axis=1)].copy()
    train, testing = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    cross_validation, test = train_test_split(testing, test_size=0.5, random_state=42, shuffle=True)

    train.to_pickle('data/processed/{}'.format(train_fn))
    test.to_pickle('data/processed/{}'.format(test_fn))
    cross_validation.to_pickle('data/processed/{}'.format(cv_fn))

if __name__ == "__main__":
    split_data('merged_data.pkl','train_data.pkl','test_data.pkl','cv_data.pkl')