import pandas as pd
from sklearn.model_selection import  train_test_split

def split_data(input_fn,train_fn, test_fn, cv_fn, test_size=.4):
    df = pd.read_pickle('data/interim/{}'.format(input_fn))
    train, testing = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    cross_validation, test = train_test_split(testing, test_size=0.5, random_state=42, shuffle=True)

    train.to_pickle('data/processed/{}'.format(train_fn))
    test.to_pickle('data/processed/{}'.format(test_fn))
    cross_validation.to_pickle('data/processed/{}'.format(cv_fn))

if __name__ == "__main__":
    split_data('merged_data.pkl','train_data.pkl','test_data.pkl','cv_data.pkl')