from matplotlib.pyplot import plot

__author__ = "Sven Köpke"
__copyright__ = "Sven Köpke 2019"
__version__ = "0.0.1"
__license__ = "MIT"

from config import *
import math
import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import accuracy_score

from DataCollection import generate_study_period, retrieve_index_history, create_constituency_matrix
from core.data_processor import DataLoader
from core.model import LSTMModel, RandomForestModel
from utils import plot_results, plot_train_val, get_most_recent_file, lookup_multiple, check_directory_for_file
from colorama import Fore, Back, Style


def main(index_id='150095', cols: list = None, force_download=False, data_only=False, last_n=None,
         load_last: bool = False,
         start_index: int = -1001, end_index: int = -1, model_type: str = 'deep_learning') -> None:
    """
    Run data preparation and model training

    :param model_type:
    :param cols: Relevant columns for model training and testing
    :param load_last: Flag indicating whether to load last model weights from storage
    :param index_id: Index identifier
    :param last_n: Flag indicating the number of days to consider; *None* in case whole data history is considered
    :param data_only: Flag indicating whether to download data only without training the model
    :param force_download: Flag indicating whether to overwrite data if index data folder already exists
    :param start_index: Index of period start date
    :param end_index: Index of period end date
    :return: None
    """

    # Load index name dict and get index name
    index_name, lookup_table = lookup_multiple(
        {'Global Dictionary':
             {'file_path': 'gvkeyx_name_dict.json',
              'lookup_table': 'comp.g_idxcst_his'},
         'North American Dictionary':
             {'file_path': 'gvkeyx_name_dict_na.json',
              'lookup_table': 'comp.idxcst_his'}},
        data_folder=ROOT_DIR, index_id=index_id)

    folder_path = os.path.join(ROOT_DIR, 'data', index_name.lower().replace(' ', '_'))  # Path to index data folder

    # Add 'return_index' column for feature generation in case of tree-based model
    if model_type == 'tree_based':
        cols.append('return_index')

    # JOB: Check whether index data already exist; create folder and set 'load_from_file' flag to false if non-existent
    load_from_file = check_directory_for_file(index_name=index_name, folder_path=folder_path,
                                              force_download=force_download)

    # JOB: Load configurations
    configs = json.load(open('config.json', 'r'))

    # JOB: Check if saved model folder exists and create one if not
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    if not load_from_file:
        # JOB: Create or load constituency matrix
        print('Creating constituency matrix ...')
        create_constituency_matrix(load_from_file=load_from_file, index_id=index_id, lookup_table=lookup_table,
                                   folder_path=folder_path)
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
        print(f'Finished downloading data for {index_name}.')
        print(f'Data set contains {data_length} individual dates.')
        print(full_data.head(10))
        return

    # JOB: Specify study period interval
    period_range = (start_index, end_index)

    # JOB: Get study period data and split index
    try:
        study_period_data, split_index = generate_study_period(constituency_matrix=constituency_matrix,
                                                               full_data=full_data,
                                                               period_range=period_range,
                                                               index_name=index_name, configs=configs, cols=cols,
                                                               folder_path=folder_path)
    except AssertionError as ae:
        print(ae)
        print('Not all constituents in full data set. Terminating execution.')
        return

    # JOB: Get all dates in study period
    full_date_range = study_period_data.index.unique()
    print(f'Study period length: {len(full_date_range)}\n')

    # JOB: Set MultiIndex to stock identifier and select relevant columns
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])[
        ['datadate', *cols]]

    # Get unique stock indices in study period
    unique_indices = study_period_data.index.unique()

    # JOB: Obtain training and test data as well as test data index
    x_train, y_train, x_test, y_test, test_data_index = preprocess_data(study_period_data=study_period_data,
                                                                        split_index=split_index,
                                                                        unique_indices=unique_indices, cols=cols,
                                                                        configs=configs,
                                                                        full_date_range=full_date_range,
                                                                        model_type=model_type)

    print(f'Length of training data: {x_train.shape}')
    print(f'Length of test data: {x_test.shape}')
    print(f'Length of test label data: {y_test.shape}')

    # Data size conformity checks
    print('\nChecking for training data size conformity: %s' % (len(x_train) == len(y_train)))
    print('Checking for test data size conformity: %s' % (len(x_test) == len(y_test)))
    print('Checking for test data index size conformity: %s \n' % (len(y_test) == len(test_data_index)))
    if len(y_test) != len(test_data_index):
        print(f'{Fore.RED}{Back.YELLOW}Lengths do not conform!{Style.RESET_ALL}')
        raise AssertionError('Test data index length is not conforming.')

    if (len(x_train) != len(y_train)) or (len(x_test) != len(y_test)):
        raise AssertionError('Data length does not conform.')

    # JOB: Determine target label distribution in train and test sets
    target_mean_train = np.mean(y_train)
    target_mean_test = np.mean(y_test)
    print(f'Average target label (training): {np.round(target_mean_train, 4)}')
    print(f'Average target label (test): {np.round(target_mean_test, 4)}\n')
    print(f'Performance validation thresholds: \n'
          f'Training: {np.round(1 - target_mean_train, 4)}\n'
          f'Testing: {np.round(1 - target_mean_test, 4)}')

    predictions = None

    if model_type == 'deep_learning':
        # JOB: Load model from storage
        if load_last:
            model = LSTMModel()
            model.load_model(get_most_recent_file('saved_models'))

        # JOB: Build model from configs
        else:
            model = LSTMModel(index_name.lower().replace(' ', '_'))
            model.build_model(configs, verbose=2)

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

    elif model_type == 'tree_based':
        model = RandomForestModel(index_name.lower().replace(' ', '_'))
        model.build_model(verbose=1)

        model.model.fit(x_train, y_train)

        predictions = model.model.predict_proba(x_test)[:, 1]
        print(predictions)

    # Create data frame with true and predicted values
    test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                       index=pd.MultiIndex.from_tuples(test_data_index, names=['datadate', 'stock_id']))

    study_period_data.index = study_period_data.index.tolist()  # Flatten MultiIndex to tuples
    study_period_data.index.name = 'stock_id'  # Rename index
    study_period_data.set_index('datadate', append=True, inplace=True)

    # JOB: Merge test set with study period data
    test_set_comparison = test_set_comparison.merge(study_period_data, how='inner', left_index=True,
                                                    right_on=['datadate', 'stock_id'])

    # JOB: Create normalized predictions
    test_set_comparison.loc[:, 'norm_prediction'] = test_set_comparison.loc[:, 'prediction'].gt(
        test_set_comparison.groupby('datadate')['prediction'].transform('median')).astype(np.int8)

    # JOB: Create cross-sectional ranking
    test_set_comparison.loc[:, 'prediction_rank'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        method='first').astype('int16')
    test_set_comparison.loc[:, 'prediction_percentile'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        pct=True)

    cross_section_size = int(round(test_set_comparison.groupby('datadate')['y_test'].count().mean()))
    print(f'Average size of cross sections: {cross_section_size}')

    # top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, int(cross_section_size / 10), int(cross_section_size / 5),
                  int(cross_section_size / 4), int(cross_section_size / 2.5),
                  int(cross_section_size / 2)]

    top_k_accuracies = pd.DataFrame({'Accuracy': []})
    top_k_accuracies.index.name = 'k'

    for top_k in top_k_list:
        # JOB: Filter test data by top/bottom k affiliation
        filtered_data = test_set_comparison[(test_set_comparison['prediction_rank'] <= top_k) | (
                test_set_comparison['prediction_rank'] > cross_section_size - top_k)]

        # print(filtered_data.sample(10))
        # print()

        # JOB: Calculate accuracy score
        if model_type == 'deep_learning':
            accuracy = binary_accuracy(filtered_data['y_test'].values,
                                       filtered_data['norm_prediction'].values).numpy()

        elif model_type == 'tree_based':
            top_k_accuracies.loc[top_k] = accuracy_score(filtered_data['y_test'].values,
                                                         filtered_data['norm_prediction'].values)

    print(top_k_accuracies)

    top_k_accuracies.plot(kind='line', legend=True, fontsize=14)
    plt.savefig(os.path.join(ROOT_DIR, folder_path, 'top_k_acc.png'), dpi=600)
    plt.show()

    # JOB: Plot training and validation metrics
    try:
        plot_train_val(history, configs['model']['metrics'], store_png=True, folder_path=folder_path)
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


def preprocess_data(study_period_data: pd.DataFrame, unique_indices: pd.MultiIndex, cols: list, split_index: int,
                    configs: dict, full_date_range: pd.Index, model_type: str) -> tuple:
    """
    Pre-process study period data to obtain training and test sets as well as test data index

    :param model_type: 'deep_learning' or 'tree_based'
    :param study_period_data: Full data for study period
    :param unique_indices: MultiIndex of unique indices in study period
    :param cols: Relevant columns
    :param split_index: Split index
    :param configs: Configuration dict
    :param full_date_range: Index of dates
    :return: Tuple of (x_train, y_train, x_test, y_test, test_data_index)
    """

    # JOB: Instantiate training and test data
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    test_data_index = pd.Index([])

    # JOB: Iterate through individual stocks and generate training and test data
    for stock_id in unique_indices:
        id_data = study_period_data.loc[stock_id].set_index('datadate')

        try:
            # JOB: Initialize DataLoader
            data = DataLoader(
                id_data, cols=cols,
                seq_len=configs['data']['sequence_length'], full_date_range=full_date_range, stock_id=stock_id,
                split_index=split_index, model_type=model_type
            )
        except AssertionError as ae:
            print(ae)
            continue

        # print('Length of data for ID %s: %d' % (stock_id, len(id_data)))

        # JOB: Generate training data
        x, y = data.get_train_data()

        # JOB: Generate test data
        x_t, y_t = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
        )

        # JOB: In case training/test set is empty, set to first batch, otherwise append data
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
                print(f'\nMismatch for index {stock_id}')
                print(f'{Fore.YELLOW}{Back.RED}{Style.BRIGHT}Lengths do not conform!{Style.RESET_ALL}')
                print(f'Target length: {len(y_t)}, index length: {len(data.data_test_index)}')
            else:
                pass

            # JOB: Append to test data index
            test_data_index = test_data_index.append(data.data_test_index)

        # print('Length of test labels: %d' % len(y_test))
        # print('Length of test index: %d\n' % len(test_data_index))

        if len(y_test) != len(test_data_index):
            raise AssertionError('Data length is not conforming.')

    return x_train, y_train, x_test, y_test, test_data_index


if __name__ == '__main__':
    # main(load_latest_model=True)
    index_list = ['150928']

    for index_id in index_list:
        main(index_id=index_id, cols=['above_cs_med', 'stand_d_return'], force_download=False,
             data_only=False,
             load_last=False, start_index=-2800,
             end_index=-1799, model_type='tree_based')

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
