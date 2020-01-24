"""
This module serves as a starting point for model training and conducting experiments with various specifications.
"""

import json
import os
from collections import deque

import pandas as pd
from colorama import Fore, Style

from matplotlib import pyplot as plt

import config
from config import ROOT_DIR
from core.analysis import StatsReport, VisTable, resample_month_end, df_to_html, DataTable
from core.data_collection import load_full_data, add_constituency_col, to_actual_index
from core.execute import main
from core.model import WeightedEnsemble, MixedEnsemble
from core.utils import get_study_period_ranges, Timer, print_study_period_ranges, CSVReader, get_run_number, \
    pretty_print_table

if __name__ == '__main__':
    # Load configurations from file
    configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))
    model_type = None
    multiple_models = None
    ensemble = None

    # Specify dict of important indices
    index_dict = {
        'DJES': '150378',  # Dow Jones European STOXX Index
        'SPEURO': '150913',  # S&P Euro Index
        'EURONEXT': '150928',  # Euronext 100 Index
        'DJ600': '150376',  # Dow Jones STOXX 600 Price Index
        'DAX': '150095'
    }

    # JOB: Specify index ID, relevant columns and study period length
    index_id = index_dict['DAX']
    cols = ['above_cs_med', *configs['data']['columns']]
    study_period_length = 1000
    verbose = 1
    plotting = False

    config.run_id = get_run_number()

    report = StatsReport()
    report.summary(last_only=False, score_list=['Top-k Annualized Sharpe'], k=10, by_model_type=True)
    # report.to_html()

    # JOB: Specify classifier

    # multiple_models = ['RandomForestClassifier']
    # ensemble = ['RandomForestClassifier', 'ExtraTreesClassifier']
    multiple_models = ['RandomForestClassifier', ['ExtraTreesClassifier', 'LSTM']]
    # multiple_models = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']

    # JOB: Calculate test_period_length from split ratio
    test_period_length = round(study_period_length * (1 - configs['data']['train_test_split']))

    # Load full index data
    constituency_matrix, full_data, index_name, folder_path = load_full_data(index_id=index_id,
                                                                             force_download=False,
                                                                             last_n=None, columns=cols.copy(),
                                                                             merge_gics=True)

    # Determine length of full data
    data_length = full_data['datadate'].drop_duplicates().size
    print(f'Total number of individual dates: {data_length}')

    # Determine date index ranges of overlapping study periods (trading periods non-overlapping)
    study_period_ranges = get_study_period_ranges(data_length=data_length, test_period_length=test_period_length,
                                                  study_period_length=study_period_length,
                                                  index_name=index_name, reverse=True, verbose=verbose)

    # JOB: Print all data ranges for study periods
    print_study_period_ranges(full_data, study_period_ranges)

    total_runtime_timer = Timer().start()

    # JOB: Iteratively fit model on all study periods
    # list(sorted(study_period_ranges.keys()))
    for study_period_ix in range(6, len(study_period_ranges) + 1):
        date_range = study_period_ranges.get(study_period_ix)
        print(f'\n\n{Fore.YELLOW}{Style.BRIGHT}Fitting on period {study_period_ix}.{Style.RESET_ALL}')
        timer = Timer().start()

        if multiple_models:
            # JOB: Run multiple models (may include ensembles)
            for model_type in multiple_models:
                if isinstance(model_type, list):

                    if MixedEnsemble.is_mixed_ensemble(model_type, configs):
                        ensemble = MixedEnsemble(index_name=index_name.lower().replace(' ', '_'),
                                                 classifier_type_list=model_type, configs=configs, verbose=verbose)
                        model_type = None

                    elif WeightedEnsemble.is_weighted_ensemble(model_type, configs):
                        ensemble = WeightedEnsemble(index_name=index_name.lower().replace(' ', '_'),
                                                    classifier_type_list=model_type, configs=configs, verbose=verbose)
                        model_type = None
                else:
                    ensemble = None
                main(index_id=index_id, index_name=index_name, full_data=full_data.copy(),
                     constituency_matrix=constituency_matrix,
                     columns=cols.copy(), folder_path=folder_path,
                     data_only=False,
                     load_last=False, start_index=date_range[0],
                     end_index=date_range[1], model_type=model_type, ensemble=ensemble, verbose=verbose,
                     plotting=plotting)

        elif ensemble:
            # JOB: Run tree-based ensemble only
            main(index_id=index_id, index_name=index_name, full_data=full_data.copy(),
                 constituency_matrix=constituency_matrix,
                 columns=cols.copy(), folder_path=folder_path,
                 data_only=False,
                 load_last=False, start_index=date_range[0],
                 end_index=date_range[1],
                 ensemble=WeightedEnsemble(index_name=index_name.lower().replace(' ', '_'),
                                           classifier_type_list=ensemble, configs=configs, verbose=verbose),
                 verbose=verbose, plotting=plotting)

        else:
            # JOB: Run single classifier
            main(index_id=index_id, index_name=index_name, full_data=full_data.copy(),
                 constituency_matrix=constituency_matrix,
                 columns=cols.copy(), folder_path=folder_path,
                 data_only=False,
                 load_last=False, start_index=date_range[0],
                 end_index=date_range[1], model_type=model_type, verbose=verbose, plotting=plotting)

        print(f'\n\n{Fore.GREEN}{Style.BRIGHT}Done fitting on period {study_period_ix}.{Style.RESET_ALL}')
        timer.stop()

    total_runtime_timer.stop()

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
