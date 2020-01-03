__author__ = "Sven Köpke"
__copyright__ = "Sven Köpke 2019"
__version__ = "0.0.1"
__license__ = "MIT"

import datetime
import json
import os
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import binary_accuracy

from DataCollection import generate_study_period, retrieve_index_history, create_constituency_matrix, create_gics_matrix
from config import *
from core.data_processor import DataLoader
from core.model import LSTMModel, RandomForestModel
from utils import plot_train_val, get_most_recent_file, lookup_multiple, check_directory_for_file, CSVWriter, \
    check_data_conformity


def main(index_id='150095', cols: list = None, force_download=False, data_only=False, last_n=None,
         load_last: bool = False, train_full=False, study_period_length=1000,
         start_index: int = -1001, end_index: int = -1, model_type: str = 'deep_learning', verbose=2) -> None:
    """
    Run data preparation and model training

    :param study_period_length:
    :param verbose:
    :param train_full:
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

    # JOB: Load configurations
    configs = json.load(open('config.json', 'r'))

    # Add 'return_index' column for feature generation in case of tree-based model
    if model_type == 'tree_based':
        cols.append('return_index')

    constituency_matrix, full_data, index_name, folder_path = load_full_data(force_download=force_download,
                                                                             last_n=last_n,
                                                                             configs=configs)

    full_data = full_data.get([column for column in full_data.columns if not column.startswith('Unnamed')])
    full_data.dropna(subset=['daily_return'], inplace=True)

    # JOB: Query number of dates in full data set
    data_length = full_data['datadate'].drop_duplicates().size  # Number of individual dates

    if data_only:
        print(f'Finished downloading data for {index_name}.')
        print(f'Data set contains {data_length} individual dates.')
        print(full_data.head(10))
        print(full_data.tail(10))
        exit(0)

    if train_full:
        n_full_periods = data_length // study_period_length
        print(f'Available index history for {index_name} allows for {n_full_periods} study periods.')
        exit()

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
    start_date = full_date_range.min().date()
    end_date = full_date_range.max().date()

    # JOB: Set MultiIndex to stock identifier and select relevant columns
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])
    # ['datadate', *cols]]

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
    print(f'Length of test target data: {y_test.shape}')

    target_mean_train, target_mean_test = check_data_conformity(x_train=x_train, y_train=y_train, x_test=x_test,
                                                                y_test=y_test, test_data_index=test_data_index)

    print(f'Average target label (training): {np.round(target_mean_train, 4)}')
    print(f'Average target label (test): {np.round(target_mean_test, 4)}\n')
    print(f'Performance validation thresholds: \n'
          f'Training: {np.round(1 - target_mean_train, 4)}\n'
          f'Testing: {np.round(1 - target_mean_test, 4)}')

    history = None
    predictions = None
    model = None
    # JOB: Test model performance on test data
    if model_type == 'deep_learning':
        # JOB: Load model from storage
        if load_last:
            model = LSTMModel()
            model.load_model(get_most_recent_file('saved_models'))

        # JOB: Build model from configs
        else:
            model = LSTMModel(index_name=index_name.lower().replace(' ', '_'))
            model.build_model(configs, verbose=2)

            # JOB: In-memory training
            history = model.train(
                x_train,
                y_train,
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir'], configs=configs, verbose=verbose
            )

        # JOB: Make point prediction
        predictions = model.predict_point_by_point(x_test)

    elif model_type == 'tree_based':
        model = RandomForestModel(index_name.lower().replace(' ', '_'))
        model.build_model(verbose=verbose)

        # JOB: Fit model
        model.model.fit(x_train, y_train)

        predictions = model.model.predict_proba(x_test)[:, 1]

    # JOB: Test model
    test_model(predictions=predictions, configs=configs, folder_path=folder_path, test_data_index=test_data_index,
               y_test=y_test, study_period_data=study_period_data, model_type=model_type, history=history,
               index_id=index_id, index_name=index_name, study_period_length=len(full_date_range), model=model,
               period_range=period_range, start_date=start_date, end_date=end_date)


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


def test_model(predictions: pd.Series, configs: dict, folder_path: str, test_data_index: pd.MultiIndex,
               y_test: np.array,
               study_period_data: pd.DataFrame, model_type: str = 'deep_learning', history=None, index_id='',
               index_name='', study_period_length: int = 0, model=None, period_range: tuple = (0, 0),
               start_date: datetime.date = datetime.date.today(), end_date: datetime.date = datetime.date.today()):
    # JOB: Create data frame with true and predicted values
    test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                       index=pd.MultiIndex.from_tuples(test_data_index, names=['datadate', 'stock_id']))

    # JOB: Transform index of study period data to match test_set_comparison index
    study_period_data.index = study_period_data.index.tolist()  # Flatten MultiIndex to tuples
    study_period_data.index.name = 'stock_id'  # Rename index
    study_period_data.set_index('datadate', append=True, inplace=True)

    # JOB: Merge test set with study period data
    test_set_comparison = test_set_comparison.merge(study_period_data, how='inner', left_index=True,
                                                    right_on=['datadate', 'stock_id'])

    # JOB: Create normalized predictions (e.g., directional prediction relative to cross-sectional median of predictions)
    test_set_comparison.loc[:, 'norm_prediction'] = test_set_comparison.loc[:, 'prediction'].gt(
        test_set_comparison.groupby('datadate')['prediction'].transform('median')).astype(np.int8)

    # JOB: Create cross-sectional ranking
    test_set_comparison.loc[:, 'prediction_rank'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        method='first', ascending=False).astype('int16')
    test_set_comparison.loc[:, 'prediction_percentile'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        pct=True)

    cross_section_size = int(round(test_set_comparison.groupby('datadate')['y_test'].count().mean()))
    print(f'Average size of cross sections: {cross_section_size}')

    # Define top k values
    top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, int(cross_section_size / 10), int(cross_section_size / 5),
                  int(cross_section_size / 4), int(cross_section_size / 2.5),
                  int(cross_section_size / 2)]

    # Create empty dataframe for top-k accuracies
    top_k_accuracies = pd.DataFrame(
        {'Accuracy': [], 'Mean Daily Return': [], 'Short Positions': [], 'Long Positions': []})
    top_k_accuracies.index.name = 'k'

    for top_k in top_k_list:
        # JOB: Filter test data by top/bottom k affiliation
        long_positions = test_set_comparison[test_set_comparison['prediction_rank'] <= top_k]
        short_positions = test_set_comparison[test_set_comparison['prediction_rank'] > cross_section_size - top_k]
        short_positions.loc[:, 'daily_return'] = - short_positions.loc[:, 'daily_return']

        full_portfolio = pd.concat([long_positions, short_positions], axis=0)

        # print(full_portfolio.head())
        # print(full_portfolio.tail())

        # print(full_portfolio.sample(10))
        accuracy = None
        mean_daily_return = None
        mean_daily_short = None
        mean_daily_long = None

        # JOB: Calculate accuracy score
        if model_type == 'deep_learning':
            accuracy = binary_accuracy(full_portfolio['y_test'].values,
                                       full_portfolio['norm_prediction'].values).numpy()
            mean_daily_return = full_portfolio['daily_return'].mean()
            mean_daily_short = short_positions['daily_return'].mean()
            mean_daily_long = long_positions['daily_return'].mean()

        elif model_type == 'tree_based':
            accuracy = accuracy_score(full_portfolio['y_test'].values,
                                      full_portfolio['norm_prediction'].values)
            mean_daily_return = full_portfolio['daily_return'].mean()
            mean_daily_short = short_positions['daily_return'].mean()
            mean_daily_long = long_positions['daily_return'].mean()

        top_k_accuracies.loc[top_k, 'Accuracy'] = accuracy
        top_k_accuracies.loc[top_k, 'Mean Daily Return'] = mean_daily_return
        top_k_accuracies.loc[top_k, 'Short Positions'] = mean_daily_short
        top_k_accuracies.loc[top_k, 'Long Positions'] = mean_daily_long

    print(top_k_accuracies)
    # Plot accuracies and save figure to file
    for col in top_k_accuracies.columns:
        top_k_accuracies[col].plot(kind='line', legend=True, fontsize=14)
        plt.savefig(os.path.join(ROOT_DIR, folder_path, f'top_k_{col.lower()}.png'), dpi=600)
        plt.show()

    # JOB: Plot training and validation metrics
    try:
        plot_train_val(history, configs['model']['metrics'], store_png=True, folder_path=folder_path)
    except AttributeError as ae:
        print(f'{Fore.RED}{Style.BRIGHT}Plotting failed.{Style.RESET_ALL}')
        # print(ae)
    except UnboundLocalError as ule:
        print(f'{Fore.RED}{Back.YELLOW}{Style.BRIGHT}Plotting failed. History has not been created.{Style.RESET_ALL}')
        # print(ule)

    # JOB: Evaluate model on full test data
    test_score = None
    if model_type == 'deep_learning':
        test_score = binary_accuracy(test_set_comparison['y_test'].values,
                                     test_set_comparison['norm_prediction'].values).numpy()

        print(f'\nTest score on full test set: {np.round(test_score, 4)}')

    elif model_type == 'tree_based':
        test_score = accuracy_score(test_set_comparison['y_test'].values,
                                    test_set_comparison['norm_prediction'].values)
        print(f'\nTest score on full test set: {np.round(test_score, 4)}')

    total_epochs = len(history.history['loss']) if history is not None else None
    data_record = {'Model Type': model_type,
                   'Index ID': index_id,
                   'Index Name': index_name,
                   'Number days': study_period_length,
                   'Test Set Size': len(y_test),
                   'Total Accuracy': test_score,
                   'Top-k Accuracy Scores': top_k_accuracies['Accuracy'].to_dict(),
                   'Top-k Mean Daily Return': top_k_accuracies['Mean Daily Return'].to_dict(),
                   'Model Configs': model.get_params(),
                   'Total Epochs': total_epochs,
                   'Period Range': period_range,
                   'Start Date': start_date,
                   'End Date': end_date
                   }

    logger = CSVWriter(output_path='training_log.csv', field_names=list(data_record.keys()))
    logger.add_line(data_record)


def load_full_data(force_download: bool, last_n: int, configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load all available records from the data for a specified index

    :param force_download: Flag indicating whether to overwrite existing data
    :param last_n: Number of last available dates to consider
    :param configs: Dict containing model and training configurations
    :return: Tuple of (constituency_matrix, full_data, index_name, folder_path)
    """
    # Load index name dict and get index name
    index_name, lookup_table = lookup_multiple(
        {'Global Dictionary':
             {'file_path': 'gvkeyx_name_dict.json',
              'lookup_table': 'global'},
         'North American Dictionary':
             {'file_path': 'gvkeyx_name_dict_na.json',
              'lookup_table': 'north_america'}}, index_id=index_id)

    folder_path = os.path.join(ROOT_DIR, 'data', index_name.lower().replace(' ', '_'))  # Path to index data folder

    # JOB: Check whether index data already exist; create folder and set 'load_from_file' flag to false if non-existent
    load_from_file = check_directory_for_file(index_name=index_name, folder_path=folder_path,
                                              force_download=force_download)

    # JOB: Check if saved model folder exists and create one if not
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    if not load_from_file:
        # JOB: Create or load constituency matrix
        print('Creating constituency matrix ...')
        create_constituency_matrix(load_from_file=load_from_file, index_id=index_id, lookup_table=lookup_table,
                                   folder_path=folder_path)
        print('Successfully created constituency matrix.')

    # JOB: Load constituency matrix
    print('Loading constituency matrix ...')
    constituency_matrix = pd.read_csv(os.path.join(folder_path, 'constituency_matrix.csv'), index_col=0, header=[0, 1],
                                      parse_dates=True)
    print('Successfully loaded constituency matrix.\n')

    # JOB: Create GICS (Global Industry Classification Standard) matrix
    gics_matrix = create_gics_matrix(index_id=index_id, index_name=index_name, lookup_table=lookup_table,
                                     load_from_file=load_from_file)

    # JOB: Load full data
    print('Retrieving full index history ...')
    full_data = retrieve_index_history(index_id=index_id, from_file=load_from_file, last_n=last_n,
                                       folder_path=folder_path, generate_dict=True)
    print('Successfully loaded index history.\n')

    if not load_from_file:
        # JOB: Merge gics matrix with full data set
        print('Merging GICS matrix with full data set.')
        full_data.set_index(['datadate', 'gvkey'], inplace=True)
        full_data = full_data.join(gics_matrix, how='inner').reset_index()

    return constituency_matrix, full_data, index_name, folder_path


if __name__ == '__main__':
    # main(load_latest_model=True)
    # index_list = ['150378']  # Dow Jones European STOXX Index
    # index_list = ['150913']  # S&P Euro Index
    # index_list = ['150928']  # Euronext 100 Index
    index_list = ['150095']  # DAX
    cols = ['above_cs_med', 'stand_d_return']
    for index_id in index_list:
        main(index_id=index_id, cols=cols, force_download=False,
             data_only=False,
             load_last=False, train_full=True, start_index=-1001,
             end_index=-1, model_type='tree_based', verbose=1)

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
