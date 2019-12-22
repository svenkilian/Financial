__author__ = "Sven Köpke"
__copyright__ = "Sven Köpke 2019"
__version__ = "0.0.1"
__license__ = "MIT"

import math
import os
import json
import sys
import numpy as np
import pandas as pd

from DataCollection import generate_study_period, retrieve_index_history, create_constituency_matrix
from core.data_processor import DataLoader
from core.model import LSTMModel
from utils import plot_results, plot_train_val


def main(index_id='150095', force_download=False, data_only=False, last_n=None):
    """
    Run data preparation and model training

    :return:
    :rtype:
    """

    gvkeyx_lookup_dict = json.load(open(os.path.join('data', 'gvkeyx_name_dict.json'), 'r'))
    index_name = gvkeyx_lookup_dict.get(index_id)
    folder_path = os.path.join('data', index_name.lower().replace(' ', '_'))

    # Check whether index data already exist; create folder and set 'load_from_file' flag to false if non-existent

    if os.path.exists(folder_path):
        if not force_download:
            load_from_file = True
            print('Loading data from %s from folder: %s' % (index_name, folder_path))
        else:
            load_from_file = False
            print('Downloading data from %s into existing folder: %s' % (index_name, folder_path))
    else:
        print('Creating folder for %s: %s' % (index_name, folder_path))
        os.mkdir(folder_path)
        load_from_file = False

    # JOB: Load configurations
    configs = json.load(open('config.json', 'r'))

    # JOB: Check if saved model folder exists and create one if not
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    if not load_from_file:
        # JOB: Create or load constituency matrix
        print('Creating constituency matrix ...')
        create_constituency_matrix(load_from_file=load_from_file, index_id=index_id, folder_path=folder_path)
        print('Successfully created constituency matrix.')

    print('Loading constituency matrix ...')
    # JOB: Load constituency matrix
    constituency_matrix = pd.read_csv(os.path.join(folder_path, 'constituency_matrix.csv'), index_col=0, header=[0, 1],
                                      parse_dates=True)
    print('Successfully loaded constituency matrix.')

    # JOB: Load full data
    print('Retrieving full index history ...')
    full_data = retrieve_index_history(index_id=index_id, from_file=load_from_file, last_n=last_n,
                                       folder_path=folder_path, generate_dict=True)
    print('Successfully loaded index history.')

    # Query number of dates in full data set
    data_length = full_data['datadate'].drop_duplicates().size  # Number of individual dates

    if data_only:
        print('Finished downloading data for %s' % index_name)
        print('Data set contains %d individual dates' % data_length)
        return None

    # JOB: Specify study period interval
    start_index = -2000
    end_index = -1200
    period_range = (start_index, end_index)

    # Get study period data
    study_period_data = generate_study_period(constituency_matrix=constituency_matrix, full_data=full_data,
                                              period_range=period_range, columns=['gvkey', 'iid', 'stand_d_return'],
                                              index_name=index_name, folder_path=folder_path)

    # Get all dates in study period
    full_date_range = study_period_data.index.unique()

    # Set MultiIndex to stock identifier and select relevant columns
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])[
        ['datadate', 'above_cs_med', 'stand_d_return', 'cshtrd']]

    # Get unique stock indices in study period
    unique_indices = study_period_data.index.unique()

    # Instantiate training and test data
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    # JOB: Iterate through individual stocks and generate training and test data
    for stock_id in unique_indices:
        id_data = study_period_data.loc[stock_id].set_index('datadate', drop=True).sort_index()
        # print(stock_id)

        # JOB: Initialize DataLoader
        data = DataLoader(
            id_data,
            configs['data']['train_test_split'], cols=['above_cs_med', 'stand_d_return', 'cshtrd'], from_csv=False,
            seq_len=configs['data']['sequence_length'], full_date_range=full_date_range
        )

        # print('Length of data for ID %s: %d' % (stock_id, len(id_data)))

        # JOB: Generate training data
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalize=False
        )

        # JOB: Generate test data
        x_t, y_t = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalize=False
        )

        # In case training set is empty, set to first batch, otherwise append data
        if x_train is None:
            x_train = x
            x_test = x_t
            y_train = y
            y_test = y_t
        else:
            if len(x) > 0:
                x_train = np.append(x_train, x, axis=0)
                y_train = np.append(y_train, y, axis=0)
            if len(x_t) > 0:
                x_test = np.append(x_test, x_t, axis=0)
                y_test = np.append(y_test, y_t, axis=0)

    # Data size conformity checks
    print('Checking for training data size conformity: %s' % (len(x_train) == len(y_train)))
    print('Checking for test data size conformity: %s' % (len(x_test) == len(y_test)))

    if (len(x_train) != len(y_train)) or (len(x_test) != len(y_test)):
        raise AssertionError('Data length does not conform.')

    # JOB: Determine target label distribution in train and test sets
    print('Average target label (training): %g' % np.mean(y_train))
    print('Average target label (test): %g' % np.mean(y_test))

    # JOB: Build model
    model = LSTMModel()
    model.build_model(configs)

    # JOB: In-memory training
    history = model.train(
        x_train,
        y_train,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir'], configs=configs, verbose=2
    )

    # JOB: Make point prediction
    predictions = model.predict_point_by_point(x_test)
    print(predictions[:5])

    # JOB: Plot training and validation metrics
    try:
        plot_train_val(history, configs['model']['metrics'])
    except AttributeError as ae:
        print('Plotting failed.')
        print(ae)

    # JOB: Evaluate model on test data
    test_scores = model.model.evaluate(x_test, y_test, verbose=1)

    # JOB: Print test scores
    print(pd.DataFrame(test_scores, index=model.model.metrics_names).T)


if __name__ == '__main__':
    # main(load_latest_model=True)
    index_list = ['150095']

    for index_id in index_list:
        main(index_id=index_id, force_download=False, data_only=False)

    """
    # Out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalize=configs['data']['normalize']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    
    """

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    #                                                configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # JOB: Make point prediction
    # predictions = model.predict_point_by_point(x_test)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
