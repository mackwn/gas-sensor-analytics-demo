import itertools
from scipy import stats
import pandas as pd
import numpy as np
import os

def merge_dat_files(out_fn):
    # Get .dat files from the raw folder
    files = os.listdir(os.path.join(os.getcwd(),'data/raw/'))
    files = [file for file in files if file.split('.')[-1] == 'dat']

    # Iterate over files and keep track of data
    ## Data did not appear to be in a standard format so needed custom processing
    data = []
    for file_name in files:
        with open(os.path.join(os.getcwd(),'data/raw',file_name)) as file:  # the a opens it in append mode
            for line in file:
                line = line.strip()
                line = line.split(' ')

                labels = [label for label in line[0].split(';')]
                line_data = [float(datum.split(':')[1]) for datum in line[1:]]
                line_data_feature_no = [str(int(datum.split(':')[0])) for datum in line[1:]]

                if len(labels) !=2: print('not two labels')
                if len(line_data_feature_no ) != 128: print('not 128 features')
                #if line_data_labels != list(range(1,129)): print('lables misordered')

                data.append([file_name.split('.')[0]]+labels+line_data)

    # Corresponses of chemical IDs from data source documentation
    chemical_names = {'1':'Ethanol', '2':'Ethylene', '3':'Ammonia', '4':'Acetaldehyde', '5':'Acetone', '6':'Toluene'}

    # Mnemonic names for the 128 features according to the documentation
    feature_names = ['s{sensor_id}_f{feature_id}'.format(sensor_id=sensor_id,feature_id=feature_id) for
                    sensor_id, feature_id in itertools.product(range(1,9),range(1,17))]

    # Create dataframe
    columns = ['Batch_ID','Gas_ID','Gas_Conc']  + feature_names
    df = pd.DataFrame(data,columns=columns)

    for k, v in chemical_names.items():
        df.loc[df.Gas_ID == k,'Gas_ID'] = v
        
    df['Gas_Conc'] = df.Gas_Conc.astype(float)
    df.to_pickle('data/interim/{}'.format(out_fn))

if __name__ == "__main__":
    merge_dat_files('merged_data.pkl')