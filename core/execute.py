__author__ = "Sven Köpke"
__copyright__ = "Sven Köpke 2019"
__version__ = "0.0.1"
__license__ = "MIT"

import datetime
import json
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import binary_accuracy

from core import utils
from core.data_collection import generate_study_period, retrieve_index_history, create_constituency_matrix, \
    create_gics_matrix, load_full_data
from config import *
from core.data_processor import DataLoader
from core.model import LSTMModel, TreeEnsemble
from core.utils import plot_train_val, get_most_recent_file, lookup_multiple, check_directory_for_file, CSVWriter, \
    check_data_conformity, annualize_metric, get_study_period_ranges, get_index_name, Timer, calc_sharpe, \
    calc_excess_returns, calc_sortino, add_to_json


def main(index_id='150095', index_name='', full_data=None, constituency_matrix: pd.DataFrame = None,
         folder_path: str = None,
         columns: list = None, data_only=False,
         load_last: bool = False,
         start_index: int = -1001, end_index: int = -1, model_type: str = 'LSTM', verbose=2):
    """
    Run data preparation and model training

    :param folder_path:
    :param constituency_matrix:
    :param full_data:
    :param index_name:
    :param verbose: Verbosity
    :param train_full: Flag indicating whether to train and test on all study periods
    :param model_type: Model type as strings: 'deep_learning' for LSTM, 'tree_based' for Random Forest
    :param columns: Relevant columns for model training and testing
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
    configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))

    model_type_reverse_dict = {child_type: parent_type for parent_type in configs['model_hierarchy'].keys() for
                               child_type in
                               configs['model_hierarchy'][parent_type]}

    parent_model_type = model_type_reverse_dict.get(model_type)

    # Add 'return_index' column for feature generation in case of tree-based model
    if parent_model_type == 'tree_based':
        columns.extend(['return_index'])
        columns.remove('stand_d_return')

    full_data = full_data.get([column for column in full_data.columns if not column.startswith('Unnamed')])
    full_data.dropna(subset=['daily_return'], inplace=True)

    # JOB: Query number of dates in full data set
    data_length = full_data['datadate'].drop_duplicates().shape[0]  # Number of individual dates

    if data_only:
        print(f'Finished downloading data for {index_name}.')
        print(f'Data set contains {data_length} individual dates.')
        # print(full_data.head(10))
        # print(full_data.tail(10))
        return full_data

    # JOB: Specify study period interval
    period_range = (start_index, end_index)

    # JOB: Get study period data and split index
    try:
        study_period_data, split_index = generate_study_period(constituency_matrix=constituency_matrix,
                                                               full_data=full_data.copy(),
                                                               period_range=period_range,
                                                               index_name=index_name, configs=configs, cols=columns,
                                                               folder_path=folder_path)
    except AssertionError as ae:
        print(ae)
        print('Not all constituents in full data set. Terminating execution.')
        return
    except RuntimeWarning as rtw:
        print(rtw)
        return

    # Delete full data copy
    del full_data

    # JOB: Get all dates in study period
    full_date_range = study_period_data.index.unique()
    print(f'Study period length: {len(full_date_range)}\n')
    start_date = full_date_range.min().date()
    end_date = full_date_range.max().date()

    # JOB: Set MultiIndex to stock identifier
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])

    # Get unique stock indices in study period
    unique_indices = study_period_data.index.unique()

    # Calculate Sharpe Ratio for full index history
    sharpe_ratio_sp = calc_sharpe(study_period_data.set_index('datadate').loc[:, ['daily_return']], annualize=True)

    print(f'Annualized Sharpe Ratio: {np.round(sharpe_ratio_sp, 4)}')

    # JOB: Obtain training and test data as well as test data index
    x_train, y_train, x_test, y_test, test_data_index = preprocess_data(study_period_data=study_period_data,
                                                                        split_index=split_index,
                                                                        unique_indices=unique_indices, cols=columns,
                                                                        configs=configs,
                                                                        full_date_range=full_date_range,
                                                                        model_type=parent_model_type)

    print(f'\nLength of training data: {x_train.shape}')
    print(f'Length of test data: {x_test.shape}')
    print(f'Length of test target data: {y_test.shape}')

    target_mean_train, target_mean_test = check_data_conformity(x_train=x_train, y_train=y_train, x_test=x_test,
                                                                y_test=y_test, test_data_index=test_data_index)

    print(f'Average target label (training): {np.round(target_mean_train, 4)}')
    print(f'Average target label (test): {np.round(target_mean_test, 4)}\n')
    print(f'Performance validation thresholds: \n'
          f'Training: {np.round(1 - target_mean_train, 4)}\n'
          f'Testing: {np.round(1 - target_mean_test, 4)}\n')

    history = None
    predictions = None
    model = None
    # JOB: Test model performance on test data
    if parent_model_type == 'deep_learning':
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

    elif parent_model_type == 'tree_based':
        model = TreeEnsemble(index_name=index_name.lower().replace(' ', '_'), model_type=model_type)
        model.build_model(configs=configs, verbose=verbose)

        # JOB: Fit model
        print(f'\n\nFitting {model_type} model ...')
        timer = Timer().start()
        model.model.fit(x_train, y_train)
        timer.stop()

        print(f'Making predictions on test set ...')
        timer = Timer().start()
        predictions = model.model.predict_proba(x_test)[:, 1]
        timer.stop()

    # JOB: Test model
    test_model(predictions=predictions, configs=configs, folder_path=folder_path, test_data_index=test_data_index,
               y_test=y_test, study_period_data=study_period_data, parent_model_type=parent_model_type,
               model_type=model_type, history=history,
               index_id=index_id, index_name=index_name, study_period_length=len(full_date_range), model=model,
               period_range=period_range, start_date=start_date, end_date=end_date)

    del study_period_data


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

    print(f'{Style.BRIGHT}{Fore.YELLOW}Pre-processing data ...{Style.RESET_ALL}')
    timer = Timer().start()

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

    print(f'{Style.BRIGHT}{Fore.YELLOW}Done pre-processing data.{Style.RESET_ALL}')
    timer.stop()

    return x_train, y_train, x_test, y_test, test_data_index


def test_model(predictions: pd.Series, configs: dict, folder_path: str, test_data_index: pd.MultiIndex,
               y_test: np.array,
               study_period_data: pd.DataFrame, parent_model_type: str = 'deep_learning', model_type: str = 'LSTM',
               history=None, index_id='',
               index_name='', study_period_length: int = 0, model=None, period_range: tuple = (0, 0),
               start_date: datetime.date = datetime.date.today(), end_date: datetime.date = datetime.date.today()):
    """
    Test model on unseen data.

    :param model_type:
    :param predictions:
    :param configs:
    :param folder_path:
    :param test_data_index:
    :param y_test:
    :param study_period_data:
    :param parent_model_type:
    :param history:
    :param index_id:
    :param index_name:
    :param study_period_length:
    :param model:
    :param period_range:
    :param start_date:
    :param end_date:
    """

    print(f'\nTesting model on unseen data ...')
    timer = Timer().start()
    # JOB: Create data frame with true and predicted values
    test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                       index=pd.MultiIndex.from_tuples(test_data_index, names=['datadate', 'stock_id']))

    # JOB: Transform index of study period data to match test_set_comparison index
    study_period_data.index = study_period_data.index.tolist()  # Flatten MultiIndex to tuples
    study_period_data.index.name = 'stock_id'  # Rename index
    study_period_data.set_index('datadate', append=True, inplace=True)

    # JOB: Merge test set with study period data
    test_set_comparison = test_set_comparison.merge(study_period_data, how='left', left_index=True,
                                                    right_on=['datadate', 'stock_id'])

    # JOB: Create normalized predictions (e.g., directional prediction relative to cross-sectional median of predictions)
    test_set_comparison.loc[:, 'norm_prediction'] = test_set_comparison.loc[:, 'prediction'].gt(
        test_set_comparison.groupby('datadate')['prediction'].transform('median')).astype(np.int8)

    # JOB: Create cross-sectional ranking
    test_set_comparison.loc[:, 'prediction_rank'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        method='first', ascending=False).astype('int16')
    test_set_comparison.loc[:, 'prediction_percentile'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        pct=True)

    test_data_start_date = test_set_comparison.index.get_level_values('datadate').min().date()
    test_data_end_date = test_set_comparison.index.get_level_values('datadate').max().date()
    test_set_n_days = test_set_comparison.index.get_level_values('datadate').unique().size
    test_set_n_constituents = test_set_comparison.index.get_level_values('stock_id').unique().size

    cross_section_size = int(round(test_set_comparison.groupby('datadate')['y_test'].count().mean()))
    print(f'Average size of cross sections: {int(cross_section_size)}')

    # Define top k values
    top_k_list = [5, 10, int(cross_section_size / 10), int(cross_section_size / 5),
                  int(cross_section_size / 4), int(cross_section_size / 2.5),
                  int(cross_section_size / 2)]

    # Create empty dataframe for top-k accuracies
    top_k_metrics = pd.DataFrame(
        {'Accuracy': [], 'Mean Daily Return': [], 'Mean Daily Return (Short)': [], 'Mean Daily Return (Long)': []})
    top_k_metrics.index.name = 'k'

    for top_k in top_k_list:
        # JOB: Filter test data by top/bottom k affiliation
        long_positions = test_set_comparison[test_set_comparison['prediction_rank'] <= top_k]
        short_positions = test_set_comparison[test_set_comparison['prediction_rank'] > cross_section_size - top_k]
        short_positions.loc[:, 'daily_return'] = - short_positions.loc[:, 'daily_return']

        full_portfolio = pd.concat([long_positions, short_positions], axis=0)

        # print(full_portfolio.head())
        # print(full_portfolio.tail())

        annualized_sharpe = calc_sharpe(full_portfolio.loc[:, ['daily_return']], annualize=True)
        annualized_sortino = calc_sortino(full_portfolio.loc[:, ['daily_return']], annualize=True)

        accuracy = None
        mean_daily_return = None
        mean_daily_short = None
        mean_daily_long = None

        # JOB: Calculate accuracy score
        if parent_model_type == 'deep_learning':
            accuracy = binary_accuracy(full_portfolio['y_test'].values,
                                       full_portfolio['norm_prediction'].values).numpy()

        elif parent_model_type == 'tree_based':
            accuracy = accuracy_score(full_portfolio['y_test'].values,
                                      full_portfolio['norm_prediction'].values)

        mean_daily_excess_return = calc_excess_returns(full_portfolio.loc[:, ['daily_return']]).mean()

        mean_daily_return = full_portfolio['daily_return'].mean()
        mean_daily_short = short_positions['daily_return'].mean()
        mean_daily_long = long_positions['daily_return'].mean()

        top_k_metrics.loc[top_k, 'Accuracy'] = accuracy
        top_k_metrics.loc[top_k, 'Mean Daily Return'] = mean_daily_return
        top_k_metrics.loc[top_k, 'Annualized Return'] = annualize_metric(mean_daily_return)
        top_k_metrics.loc[top_k, 'Mean Daily Excess Return'] = mean_daily_excess_return
        top_k_metrics.loc[top_k, 'Annualized Excess Return'] = annualize_metric(mean_daily_excess_return)
        top_k_metrics.loc[top_k, 'Annualized Sharpe'] = annualized_sharpe
        top_k_metrics.loc[top_k, 'Annualized Sortino'] = annualized_sortino
        top_k_metrics.loc[top_k, 'Mean Daily Return (Short)'] = mean_daily_short
        top_k_metrics.loc[top_k, 'Mean Daily Return (Long)'] = mean_daily_long

    # JOB: Display top-k metrics
    print(top_k_metrics)

    # Plot accuracies and save figure to file
    for col in top_k_metrics.columns:
        top_k_metrics[col].plot(kind='line', legend=True, fontsize=14)
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
    if parent_model_type == 'deep_learning':
        test_score = binary_accuracy(test_set_comparison['y_test'].values,
                                     test_set_comparison['norm_prediction'].values).numpy()

        print(f'\nTest score on full test set: {float(np.round(test_score, 4))}')

    elif parent_model_type == 'tree_based':
        test_score = accuracy_score(test_set_comparison['y_test'].values,
                                    test_set_comparison['norm_prediction'].values)
        print(f'\nTest score on full test set: {np.round(test_score, 4)}')

    total_epochs = len(history.history['loss']) if history is not None else None
    data_record = {'Experiment Run End': datetime.datetime.now().isoformat(),
                   'Parent Model Type': parent_model_type,
                   'Model Type': model_type,
                   'Index ID': index_id,
                   'Index Name': index_name,
                   'Study Period Length': study_period_length,
                   'Test Set Size': y_test.shape[0],
                   'Days Test Set': test_set_n_days,
                   'Constituent Number': test_set_n_constituents,
                   'Average Cross Section Size': cross_section_size,
                   'Test Set Start Date': test_data_start_date.isoformat(),
                   'Test Set End Date': test_data_end_date.isoformat(),
                   'Total Accuracy': float(test_score),
                   'Top-k Accuracy Scores': top_k_metrics['Accuracy'].to_dict(),
                   'Top-k Mean Daily Return': top_k_metrics['Mean Daily Return'].to_dict(),
                   'Top-k Mean Daily Excess Return': top_k_metrics['Mean Daily Excess Return'].to_dict(),
                   'Annualized Excess Return': top_k_metrics['Annualized Excess Return'].to_dict(),
                   'Top-k Annualized Return': top_k_metrics['Annualized Return'].to_dict(),
                   'Top-k Annualized Sharpe': top_k_metrics['Annualized Sharpe'].to_dict(),
                   'Top-k Annualized Sortino': top_k_metrics['Annualized Sortino'].to_dict(),
                   'Model Configs': model.get_params(),
                   'Total Epochs': total_epochs,
                   'Period Range': period_range,
                   'Study Period Start Date': start_date.isoformat(),
                   'Study Period End Date': end_date.isoformat()
                   }

    logger = CSVWriter(output_path=os.path.join(ROOT_DIR, 'data', 'training_log.csv'),
                       field_names=list(data_record.keys()))
    logger.add_line(data_record)

    add_to_json(data_record, 'data/training_log.json')

    print('Done testing on unseen data.')
    timer.stop()


if __name__ == '__main__':
    print(f'{Fore.YELLOW}{Style.BRIGHT}This module has no action on execution as \'main\'.{Style.RESET_ALL}')
