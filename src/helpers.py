import pandas as pd
import numpy as np

def df_to_arrays(df):
    cols = df.columns.values
    batch_id = data[:,0]
    y = data[:,1:3]
    X = data[:,3:]
    arrays = {
        'id':batch_id,
        'y':y,
        'X':X
    }
    return arrays