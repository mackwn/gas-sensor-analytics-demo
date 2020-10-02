import requests
import zipfile

def download_data(data_fn='raw_data'):
    # Request the file 
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip'
    # Unzip the file
    r = requests.get(url, allow_redirects=True)
    open('data/raw/{}.zip'.format(data_fn), 'wb').write(r.content)
    # Create a ZipFile Object and load sample.zip in it
    #with ZipFile('sampleDir.zip', 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        #zipObj.extractall()

def unzip_data(data_fn='raw_data'):

    with zipfile.ZipFile('data/raw/{}.zip'.format(data_fn), 'r') as zipObj:
        zipObj.extractall('data/raw/')

if __name__ == "__main__":
    
    data_fn = 'data'
    #download_data(data_fn)
    unzip_data(data_fn)