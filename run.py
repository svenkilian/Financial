"""
This module serves as a starting point for model training and conducting experiments with various specifications.
"""

import json

from config import ROOT_DIR
from core.data_collection import load_full_data
from core.utils import get_index_name, get_study_period_ranges, Timer, deannualize, calc_sharpe
from core.execute import main
from colorama import Fore, Style
import pandas as pd
import os

if __name__ == '__main__':
    # JOB: Load configurations from file
    configs = json.load(open('config.json', 'r'))

    rf_rate_series = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'rf_rate_euro.csv'), parse_dates=True,
                                 index_col=0) / 100

    # Specify dict of important indices:
    index_dict = {
        'DJES': '150378',  # Dow Jones European STOXX Index
        'SPEURO': '150913',  # S&P Euro Index
        'EURONEXT': '150928',  # Euronext 100 Index
        'DJ600': '150376',  # Dow Jones STOXX 600 Price Index
        'DAX': '150095'
    }

    # JOB: Specify index ID, relevant columns and study period length
    index_id = index_dict['DJ600']
    cols = ['above_cs_med', 'stand_d_return']
    study_period_length = 1000

    # JOB: Specify classifier
    model_type = None
    multiple_models = None
    model_type = 'ExtraTreesClassifier'
    # model_type = 'LSTM'
    # multiple_models = ['LSTM', 'RandomForestClassifier', 'ExtraTreesClassifier']
    # multiple_models = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']

    # JOB: Calculate test_period_length from split ratio
    test_period_length = round(study_period_length * (1 - configs['data']['train_test_split']))

    # Load full index data
    constituency_matrix, full_data, index_name, folder_path = load_full_data(index_id=index_id,
                                                                             force_download=False,
                                                                             last_n=None,
                                                                             merge_gics=True)

    # Determine length of full data
    data_length = full_data['datadate'].drop_duplicates().size
    print(f'Total number of individual dates: {data_length}')

    # Determine date index ranges of overlapping study periods (trading periods non-overlapping)
    study_period_ranges = get_study_period_ranges(data_length=data_length, test_period_length=test_period_length,
                                                  study_period_length=study_period_length,
                                                  index_name=index_name, reverse=True, verbose=1)

    # JOB: Print all data ranges for study periods
    full_date_range = full_data['datadate'].unique()
    print('\nStudy period date index ranges:')
    for study_period_no in list(sorted(study_period_ranges.keys())):
        date_index_tuple = study_period_ranges.get(study_period_no)
        date_tuple = tuple(
            str(pd.to_datetime(date_full).date()) for date_full in full_date_range[list(date_index_tuple)])
        print(
            f'Period {Style.BRIGHT}{Fore.YELLOW}{study_period_no}{Style.RESET_ALL}: {date_index_tuple} -> {date_tuple}')

    # JOB: Iteratively fit model on all study periods
    # list(sorted(study_period_ranges.keys()))
    for study_period_ix in range(6, len(study_period_ranges) + 1):
        date_range = study_period_ranges.get(study_period_ix)
        print(f'\n\n{Fore.YELLOW}{Style.BRIGHT}Fitting on period {study_period_ix}.{Style.RESET_ALL}')
        timer = Timer().start()

        if multiple_models:
            for model_type in multiple_models:
                main(index_id=index_id, index_name=index_name, full_data=full_data.copy(),
                     constituency_matrix=constituency_matrix,
                     columns=cols.copy(), folder_path=folder_path,
                     data_only=False,
                     load_last=False, start_index=date_range[0],
                     end_index=date_range[1], model_type=model_type, verbose=2)

        else:
            main(index_id=index_id, index_name=index_name, full_data=full_data.copy(),
                 constituency_matrix=constituency_matrix,
                 columns=cols.copy(), folder_path=folder_path,
                 data_only=False,
                 load_last=False, start_index=date_range[0],
                 end_index=date_range[1], model_type=model_type, verbose=2)

        print(f'\n\n{Fore.GREEN}{Style.BRIGHT}Done fitting on period {study_period_ix}.{Style.RESET_ALL}')
        timer.stop()

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
