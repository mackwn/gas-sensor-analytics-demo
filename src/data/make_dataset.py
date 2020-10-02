# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import sys
sys.path.append(".")
from src.data import download_data
from src.data import merge_files
from src.data import split_data


#@click.command()
#@click.argument('output_filepath', type=click.Path())
def main(output_filename='merged_data'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading data')
    download_data.download_data()
    logger.info('Unzipping files')
    download_data.unzip_data()
    logger.info('Merge raw data files into data frame')
    merge_files.merge_dat_files(output_filename)
    logger.info('Split data into test and training sets')
    split_data.split_data(output_filename)





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
