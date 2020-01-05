import json

from core.data_collection import load_full_data
from core.utils import get_index_name, get_study_period_ranges, Timer
from core.execute import main
from colorama import Fore, Style
import pandas as pd

if __name__ == '__main__':
    configs = json.load(open('config.json', 'r'))

    # main(load_latest_model=True)
    # index_list = ['150378']  # Dow Jones European STOXX Index
    # index_list = ['150913']  # S&P Euro Index
    # index_list = ['150928']  # Euronext 100 Index

    # Specify index ID, relevant columns and study period length
    index_id = '150376'
    cols = ['above_cs_med', 'stand_d_return']
    study_period_length = 1000
    # model_type = 'RandomForestClassifier'
    model_type = 'RandomForestClassifier'

    multiple_models = ['ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']

    # Calculate test_period_length from split ratio
    test_period_length = round(study_period_length * (1 - configs['data']['train_test_split']))

    # Determine number of available distinct dates in full data
    constituency_matrix, full_data, index_name, folder_path = load_full_data(index_id=index_id,
                                                                             force_download=False,
                                                                             last_n=None)

    data_length = full_data['datadate'].drop_duplicates().size
    print(f'Total number of individual dates: {data_length}')
    print(full_data.head(5))

    # Determine date index ranges of overlapping study periods (trading periods non-overlapping)
    study_period_ranges = get_study_period_ranges(data_length=data_length, test_period_length=test_period_length,
                                                  study_period_length=study_period_length,
                                                  index_name=index_name, reverse=True, verbose=1)

    full_date_range = full_data['datadate'].unique()
    print('\nStudy period date index ranges:')
    for study_period_no in list(sorted(study_period_ranges.keys())):
        date_index_tuple = study_period_ranges.get(study_period_no)
        date_tuple = tuple(
            str(pd.to_datetime(date_full).date()) for date_full in full_date_range[list(date_index_tuple)])
        print(f'Period {study_period_no}: {date_index_tuple} -> {date_tuple}')

    # JOB: Iteratively fit model on all study periods
    # list(sorted(study_period_ranges.keys()))
    for study_period_ix in range(8, len(study_period_ranges) + 1):
        date_range = study_period_ranges.get(study_period_ix)
        print(f'\n\n{Fore.YELLOW}{Style.BRIGHT}Fitting on period {study_period_ix}.{Style.RESET_ALL}')
        timer = Timer().start()

        if multiple_models:
            for model_type in multiple_models:
                main(index_id=index_id, full_data=full_data.copy(), constituency_matrix=constituency_matrix,
                     columns=cols.copy(), folder_path=folder_path,
                     data_only=False,
                     load_last=False, start_index=date_range[0],
                     end_index=date_range[1], model_type=model_type, verbose=1)

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
