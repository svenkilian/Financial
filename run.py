from matplotlib.pyplot import plot

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
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import binary_accuracy

from DataCollection import generate_study_period, retrieve_index_history, create_constituency_matrix
from core.data_processor import DataLoader
from core.model import LSTMModel
from utils import plot_results, plot_train_val, get_most_recent_file
from colorama import Fore, Back, Style


def main(index_id='150095', force_download=False, data_only=False, last_n=None, load_last: bool = False,
         start_index: int = -1001, end_index: int = -1) -> None:
    """
    Run data preparation and model training

    :param start_index: Index of period start date
    :param end_index: Index of period end date
    :return: None
    """

    gvkeyx_lookup_dict = json.load(open(os.path.join('data', 'gvkeyx_name_dict.json'), 'r'))
    index_name = gvkeyx_lookup_dict.get(index_id)
    folder_path = os.path.join('data', index_name.lower().replace(' ', '_'))

    # JOB: Check whether index data already exist; create folder and set 'load_from_file' flag to false if non-existent

    if os.path.exists(folder_path):
        if not force_download:
            load_from_file = True
            print('Loading data from %s from folder: %s \n' % (index_name, folder_path))
        else:
            load_from_file = False
            print('Downloading data from %s into existing folder: %s \n' % (index_name, folder_path))
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
    print('Successfully loaded constituency matrix.\n')

    # JOB: Load full data
    print('Retrieving full index history ...')
    full_data = retrieve_index_history(index_id=index_id, from_file=load_from_file, last_n=last_n,
                                       folder_path=folder_path, generate_dict=True)
    print('Successfully loaded index history.\n')

    # JOB: Query number of dates in full data set
    data_length = full_data['datadate'].drop_duplicates().size  # Number of individual dates

    if data_only:
        print('Finished downloading data for %s' % index_name)
        print('Data set contains %d individual dates' % data_length)
        print(full_data.head(100))
        return None

    # JOB: Specify study period interval
    period_range = (start_index, end_index)

    # Get study period data
    study_period_data = generate_study_period(constituency_matrix=constituency_matrix, full_data=full_data,
                                              period_range=period_range,
                                              index_name=index_name, folder_path=folder_path)

    # Get all dates in study period
    full_date_range = study_period_data.index.unique()
    print(f'Study period length: {len(full_date_range)}')

    # Set MultiIndex to stock identifier and select relevant columns
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])[
        ['datadate', 'above_cs_med', 'stand_d_return']]

    # Get unique stock indices in study period
    unique_indices = study_period_data.index.unique()

    # JOB: Instantiate training and test data
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    test_data_index = pd.Index([])

    # JOB: Iterate through individual stocks and generate training and test data
    for stock_id in unique_indices:
        id_data = study_period_data.loc[stock_id].set_index('datadate')
        # print(stock_id)

        # JOB: Initialize DataLoader
        data = DataLoader(
            id_data,
            split=configs['data']['train_test_split'], cols=['above_cs_med', 'stand_d_return'],
            from_csv=False,
            seq_len=configs['data']['sequence_length'], full_date_range=full_date_range, stock_id=stock_id
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
        if (len(x) > 0) and (len(y) > 0):
            if x_train is None:
                x_train = x
                y_train = y
            else:
                x_train = np.append(x_train, x, axis=0)
                y_train = np.append(y_train, y, axis=0)

        if (len(x_t) > 0) and (len(y_t) > 0):
            if x_test is None:
                x_test = x_t
                y_test = y_t
            else:
                x_test = np.append(x_test, x_t, axis=0)
                y_test = np.append(y_test, y_t, axis=0)

            if len(y_t) != len(data.data_test_index):
                print(f'{Fore.RED}{Back.YELLOW}Lengths do not conform!{Style.RESET_ALL}')

            # Append to index
            test_data_index = test_data_index.append(data.data_test_index)

        # print('Length of test labels: %d' % len(y_test))
        # print('Length of test index: %d\n' % len(test_data_index))

        if len(y_test) != len(test_data_index):
            raise Exception('Data length is not conforming.')

    # Data size conformity checks
    print('Checking for training data size conformity: %s' % (len(x_train) == len(y_train)))
    print('Checking for test data size conformity: %s' % (len(x_test) == len(y_test)))
    print('Checking for test data index conformity: %s \n' % (len(y_test) == len(test_data_index)))
    if len(y_test) != len(test_data_index):
        # raise Exception('Test data index length is inconsistent.')
        print(f'{Fore.RED}{Back.YELLOW}Lengths do not conform!{Style.RESET_ALL}')

    if (len(x_train) != len(y_train)) or (len(x_test) != len(y_test)):
        raise AssertionError('Data length does not conform.')

    # JOB: Determine target label distribution in train and test sets
    print('Average target label (training): %g' % np.mean(y_train))
    print('Average target label (test): %g \n' % np.mean(y_test))

    if load_last:
        model = LSTMModel()
        model.load_model(get_most_recent_file('saved_models'))

    else:
        # JOB: Build model
        model = LSTMModel(index_name.lower().replace(' ', '_'))
        model.build_model(configs)

        # JOB: In-memory training
        history = model.train(
            x_train,
            y_train,
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            save_dir=configs['model']['save_dir'], configs=configs, verbose=2
        )

    # JOB: Make point prediction and join with target values
    predictions = model.predict_point_by_point(x_test)

    test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                       index=pd.MultiIndex.from_tuples(test_data_index, names=['datadate', 'stock_id']))

    study_period_data.index = study_period_data.index.tolist()
    study_period_data.index.name = 'stock_id'
    study_period_data.set_index('datadate', append=True, inplace=True)

    test_set_comparison = test_set_comparison.merge(study_period_data, how='inner', left_index=True,
                                                    right_on=['datadate', 'stock_id'])

    # JOB: Create normalized predictions
    test_set_comparison.loc[:, 'norm_prediction'] = test_set_comparison.loc[:, 'prediction'].gt(
        test_set_comparison.groupby('datadate')['prediction'].transform('median')).astype(int)

    # JOB: Create cross-sectional ranking
    test_set_comparison.loc[:, 'prediction_rank'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        method='first').astype('int8')
    test_set_comparison.loc[:, 'prediction_percentile'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        pct=True)

    cross_section_size = round(test_set_comparison.groupby('datadate')['y_test'].count().mean())
    print('Average size of cross sections: %d' % cross_section_size)
    top_percentage = 0.1
    # top_k = round(top_percentage * cross_section_size)

    # top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    top_k_accuracies = pd.DataFrame({'Accuracy': []})

    for top_k in top_k_list:
        print('k = %d' % top_k)

        # JOB: Filter test data by top/bottom k affiliation
        filtered = test_set_comparison[(test_set_comparison['prediction_rank'] <= top_k) | (
                test_set_comparison['prediction_rank'] > cross_section_size - top_k)]

        # print(filtered.sample(10))
        # print()

        # JOB: Calculate accuracy score
        accuracy = binary_accuracy(filtered['y_test'].values,
                                   filtered['norm_prediction'].values).numpy()

        # print('Top-%d Accuracy: %g\n' % (top_k, round(accuracy, 4)))

        top_k_accuracies.loc[top_k] = accuracy

    top_k_accuracies.index.name = 'k'
    print(top_k_accuracies)
    top_k_accuracies.plot(kind='line', legend=True, fontsize=14)
    plt.show()

    # JOB: Plot training and validation metrics
    try:
        plot_train_val(history, configs['model']['metrics'])
    except AttributeError as ae:
        print(f'{Fore.RED}{Back.YELLOW}{Style.BRIGHT}Plotting failed.{Style.RESET_ALL}')
        # print(ae)
    except UnboundLocalError as ule:
        print(f'{Fore.RED}{Back.YELLOW}{Style.BRIGHT}Plotting failed. History has not been created.{Style.RESET_ALL}')
        # print(ule)

    # # JOB: Evaluate model on test data
    # test_scores = model.model.evaluate(x_test, y_test, verbose=2)
    #
    # # JOB: Print test scores
    # print('\nTest scores:')
    # print(pd.DataFrame(test_scores, index=model.model.metrics_names).T)


if __name__ == '__main__':
    # main(load_latest_model=True)
    index_list = ['150095']

    for index_id in index_list:
        main(index_id=index_id, force_download=False, data_only=False, load_last=False, start_index=-7000,
             end_index=-5999)

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
