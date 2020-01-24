"""
This module implements methods to collect financial data from Wharton Research Services via the wrds package
"""

import datetime
# Imports
import json
import os
import re
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import wrds
from colorama import Fore, Back, Style

# Configurations for displaying DataFrames
from config import ROOT_DIR
from core.utils import get_index_name, check_directory_for_file, Timer, lookup_multiple

pd.set_option('precision', 4)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 40)
pd.set_option('max_colwidth', 60)
pd.set_option('mode.sim_interactive', True)
pd.set_option('expand_frame_repr', True)
pd.set_option('large_repr', 'truncate')

pd.set_option('colheader_justify', 'left')
pd.set_option('display.width', 800)
pd.set_option('display.html.table_schema', False)

from sklearn.preprocessing import StandardScaler

pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def retrieve_index_history(index_id: str = None, from_file=False, last_n: int = None,
                           folder_path: str = '', generate_dict=False) -> pd.DataFrame:
    """
    Download complete daily index history and return as Data Frame (no date index)

    :return: DataFrame containing full index constituent data over full index history
    :rtype: pd.DataFrame
    """

    if not from_file:
        # Load GVKEYX lookup dict
        with open(os.path.join(ROOT_DIR, 'data', 'gvkeyx_name_dict.json'), 'r') as fp:
            gvkeyx_lookup = json.load(fp)

        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')

        # Retrieve list of all stocks (gvkeys) for specified index including full date range of historic index data
        gvkey_list, relevant_date_range = get_all_constituents(
            constituency_matrix=pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'constituency_matrix.csv'), index_col=0,
                                            header=[0, 1],
                                            parse_dates=True))

        # Set start and end date
        if last_n:
            start_date = str(relevant_date_range[-last_n].date())
        else:
            start_date = str(relevant_date_range[0].date())

        end_date = str(relevant_date_range[-1].date())

        # Specify list of companies and start and end date of query
        parameters = {'company_codes': tuple(gvkey_list), 'start_date': start_date, 'end_date': end_date}

        print('Querying full index history for index %s \n'
              'between %s and %s ...' % (gvkeyx_lookup.get(index_id), start_date, end_date))
        start_time = time.time()
        data = get_data_table(db, sql_query=True,
                              query_string="select datadate, gvkey, iid, trfd, ajexdi, cshtrd, prccd, divd, conm, curcdd, sedol, exchg, gsubind "
                                           "from comp.g_secd "
                                           "where gvkey in %(company_codes)s and datadate between %(start_date)s and %(end_date)s "
                                           "order by datadate asc",
                              index_col=['datadate', 'gvkey', 'iid'], table_info=1, params=parameters)

        end_time = time.time()

        print('Query duration: %g seconds' % (round(end_time - start_time, 2)))
        print('Number of observations: %s' % data.shape[0])
        print('Number of individual dates: %d' % data.index.get_level_values('datadate').drop_duplicates().size)

        # JOB: Add return_index and daily_return columns
        data = calculate_daily_return(data, save_to_file=False)

        # Reset index
        data.reset_index(inplace=True)

        # Save to file
        data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'))

    else:
        data = pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'),
                           dtype={'gvkey': str, 'gsubind': str, 'datadate': str, 'gics_sector': str},
                           parse_dates=False, index_col=0)
        data.loc[:, 'datadate'] = pd.to_datetime(data.loc[:, 'datadate'], infer_datetime_format=True).dt.date

    if not check_directory_for_file(index_id=index_id, folder_path=os.path.join(folder_path, 'gvkey_name_dict.json'),
                                    create_dir=False, print_status=False):
        generate_company_lookup_dict(folder_path=folder_path, data=data)

    return data


def load_full_data(index_id: str = '150095', force_download: bool = False, last_n: int = None, columns: list = None,
                   merge_gics=False) -> Tuple[
    pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load all available records from the data for a specified index

    :param columns:
    :param merge_gics:
    :param index_id: Index ID to load data for
    :param force_download: Flag indicating whether to overwrite existing data
    :param last_n: Number of last available dates to consider
    :return: Tuple of (constituency_matrix, full_data, index_name, folder_path)
    """
    # Load index name dict and get index name
    index_name, lookup_table = get_index_name(index_id=index_id)

    configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))

    folder_path = os.path.join(ROOT_DIR, 'data', index_name.lower().replace(' ', '_'))  # Path to index data folder

    # JOB: Check whether index data already exist; create folder and set 'load_from_file' flag to false if non-existent
    load_from_file = check_directory_for_file(index_name=index_name, folder_path=folder_path,
                                              force_download=force_download)

    # JOB: Check if saved model folder exists and create one if not
    if not os.path.exists(os.path.join(ROOT_DIR, configs['model']['save_dir'])):
        os.makedirs(configs['model']['save_dir'])

    if not load_from_file:
        # JOB: Create or load constituency matrix
        print('Creating constituency matrix ...')
        timer = Timer()
        timer.start()
        create_constituency_matrix(load_from_file=load_from_file, index_id=index_id, lookup_table=lookup_table,
                                   folder_path=folder_path)
        print('Successfully created constituency matrix.')
        timer.stop()

    # JOB: Load constituency matrix
    print('Loading constituency matrix ...')
    constituency_matrix = pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'constituency_matrix.csv'), index_col=0,
                                      header=[0, 1],
                                      parse_dates=True)
    print('Successfully loaded constituency matrix.\n')

    # JOB: Load full data
    print('Retrieving full index history ...')
    timer = Timer().start()
    full_data = retrieve_index_history(index_id=index_id, from_file=load_from_file, last_n=last_n,
                                       folder_path=folder_path, generate_dict=False)
    full_data.set_index('datadate', inplace=True)
    # JOB: Sort by date and reset index
    full_data.sort_index(inplace=True)
    full_data.reset_index(inplace=True)

    if merge_gics and 'gsubind' not in full_data.columns:
        # JOB: Create GICS (Global Industry Classification Standard) matrix
        gics_matrix = create_gics_matrix(index_id=index_id, index_name=index_name, lookup_table=lookup_table,
                                         load_from_file=load_from_file)
        # JOB: Merge gics matrix with full data set
        print('Merging GICS matrix with full data set.')
        full_data.set_index(['datadate', 'gvkey'], inplace=True)
        full_data = full_data.join(gics_matrix, how='left').reset_index()
        # Save to file
        full_data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'))

    if 'gics_sector' not in full_data.columns:
        # JOB: Extract 2-digit GICS code
        generate_gics_sector(full_data)
        # Save to file
        print('Saving modified data to file ...')
        full_data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'))

    if 'gsector' in full_data.columns:
        full_data.drop(columns=['gsector'], inplace=True)
        # Save to file
        print('Saving modified data to file ...')
        full_data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'))

    if 'ind_mom_ratio' in columns:
        # JOB: Add 6-month (=120 days) momentum column
        print('Adding 6-month momentum ...')
        timer = Timer().start()
        full_data.set_index(keys=['datadate', 'gvkey', 'iid'], inplace=True)
        full_data.loc[:, '6m_mom'] = full_data.groupby(level=['gvkey', 'iid'])['return_index'].apply(
            lambda x: x.pct_change(periods=120))
        timer.stop()

        full_data.reset_index(inplace=True)

    gics_map = pd.read_json(os.path.join(ROOT_DIR, 'data', 'gics_code_dict.json'), orient='records',
                            typ='series').rename('gics_sec')
    gics_map = {str(key): val for key, val in gics_map.to_dict().items()}
    full_data['gics_sec'] = full_data.loc[:, 'gics_sector'].replace(gics_map, inplace=True)

    print('Successfully loaded index history.')
    timer.stop()
    print()

    # Drop records with missing daily_return variable
    full_data.dropna(subset=['daily_return'], inplace=True)

    return constituency_matrix, full_data, index_name, folder_path


def generate_study_period(constituency_matrix: pd.DataFrame, full_data: pd.DataFrame,
                          period_range: tuple, index_name: str, configs: dict, folder_path: str, columns=None) -> (
        pd.DataFrame, int):
    """
    Generate a time-period sample for a study period

    :param columns: Feature columns
    :param configs: Dictionary containing model and training configurations
    :param folder_path: Path to data folder
    :param period_range: Date index range of study period in form (start_index, end_index)
    :type period_range: tuple
    :param full_data: Full stock data
    :type full_data: pd.DataFrame
    :param constituency_matrix: Constituency matrix
    :type constituency_matrix: pd.DataFrame
    :param index_name: Name of index
    :type index_name: str

    :return: Tuple Study period sample and split index
    :rtype: tuple[pd.DataFrame, int]
    """

    # Convert date columns to DatetimeIndex
    full_data.loc[:, 'datadate'] = pd.to_datetime(full_data['datadate'])

    # Set index to date column
    full_data.set_index('datadate', inplace=True)

    # Get unique dates
    unique_dates = full_data.index.drop_duplicates()
    split_ratio = configs['data']['train_test_split']
    i_split = len(unique_dates[:period_range[0]]) + int(
        len(unique_dates[period_range[0]:period_range[1]]) * split_ratio)
    split_date = unique_dates[i_split]

    # Detect potential out-of-bounds indices
    if abs(period_range[0]) > len(unique_dates) or abs(period_range[1] > len(unique_dates)):
        print(f'Index length is {len(unique_dates)}. Study period range ist {period_range}.')
        raise IndexError('Period index out of bounds.')

    print(f'Retrieving index constituency for {index_name} as of {split_date.date()}.')
    try:
        constituent_indices = get_index_constituents(constituency_matrix, date=split_date,
                                                     folder_path=folder_path, print_constituents=False)
        if len(constituent_indices) < 2:
            raise RuntimeWarning('Period contains too few constituents. Continuing with next study period.')
    except IndexError as ie:
        print(
            f'{Fore.RED}{Back.YELLOW}{Style.BRIGHT}Period index out of bounds. Choose different study period bounds.'
            f'{Style.RESET_ALL}')
        print(', '.join(ie.args))
        print('Terminating program.')
        sys.exit(1)

    full_data.reset_index(inplace=True)

    # JOB: Select relevant data
    # Select relevant stocks
    full_data = full_data.set_index(['gvkey', 'iid'])
    # print(f'Length of intersection: {len(constituent_indices.intersection(full_data.index))}')
    # print(f'Length of difference: {len(constituent_indices.difference(full_data.index))}')
    assert len(constituent_indices.intersection(full_data.index)) == len(constituent_indices)

    print('\nFiltering by study period constituents ...')
    timer = Timer().start()
    full_data = full_data.loc[constituent_indices, :]
    # print(f'Number of constituents (deduplicated): {full_data.index.get_level_values(level="gvkey").unique().shape[0]}')
    full_data.reset_index(inplace=True)
    full_data.set_index('datadate', inplace=True)
    full_data.sort_index(inplace=True)
    timer.stop()

    # JOB: Select data from study period
    print(f'Retrieving data from {unique_dates[period_range[0]].date()} to {unique_dates[period_range[1]].date()} \n')
    study_data = full_data.loc[unique_dates[period_range[0]]:unique_dates[period_range[1]]]

    if 'ind_mom_ratio' in columns:
        # JOB: Drop records without 6 month-returns
        print('Removing records without 6-month momentum ...')
        print('Missing records before removal:')
        print(study_data[study_data['6m_mom'].isna()].set_index(['gvkey', 'iid'], append=True).index.get_level_values(
            level='gvkey').unique().shape[0])
        study_data.dropna(how='any', subset=['6m_mom'], inplace=True)
        print('Missing records after removal:')
        print(study_data[study_data['6m_mom'].isna()].set_index(['gvkey', 'iid'], append=True).index.get_level_values(
            level='gvkey').unique().shape[0])

    del full_data

    print(f'Study period length: {len(study_data.index.unique())}')

    study_data_split_index = study_data.index.unique().get_loc(split_date, method='ffill')
    study_data_split_date = study_data.index.unique()[study_data_split_index]
    print(f'Study data split index: {study_data_split_index}')
    print(f'Study data split date: {study_data_split_date.date()}')

    # JOB: Calculate mean and standard deviation of daily returns for training period
    mean_daily_return = study_data.loc[unique_dates[period_range[0]]:split_date, 'daily_return'].mean()
    std_daily_return = study_data.loc[unique_dates[period_range[0]]:split_date, 'daily_return'].std()
    print('Mean daily return: %g' % mean_daily_return)
    print('Std. daily return: %g \n' % std_daily_return)

    # JOB: Calculate standardized daily returns
    study_data.loc[:, 'stand_d_return'] = (study_data['daily_return'] - mean_daily_return) / std_daily_return

    # JOB: Drop observations with critical n/a values
    # Review: Check whether operation is redundant
    # critical_cols = ['daily_return']
    # study_data.dropna(how='any', subset=critical_cols, inplace=True)

    # JOB: Create target
    study_data.loc[:, 'above_cs_med'] = study_data['daily_return'].gt(
        study_data.groupby('datadate')['daily_return'].transform('median')).astype(np.int8)
    study_data.loc[:, 'cs_med'] = study_data.groupby('datadate')['daily_return'].transform('median')

    # JOB: Create cross-sectional ranking
    # study_data.loc[:, 'cs_rank'] = study_data.groupby('datadate')['daily_return'].rank(method='first', ascending=False).astype('int16')
    # study_data.loc[:, 'cs_percentile'] = study_data.groupby('datadate')['daily_return'].rank(pct=True)

    # JOB: Add columns with number of securities in cross-section
    study_data.loc[:, 'cs_length'] = study_data.groupby('datadate')['daily_return'].count()
    study_data.reset_index(inplace=True)

    study_data.reset_index(inplace=True)
    study_data.set_index('datadate', inplace=True)

    if 'ind_mom_ratio' in columns:
        # JOB: Add industry momentum column
        study_data.loc[:, 'ind_mom'] = study_data.groupby(['gics_sector', 'datadate'])['6m_mom'].transform('mean')

        # Drop observations with missing industry momentum
        study_data.dropna(how='any', subset=['ind_mom'], inplace=True)

        # JOB: Add 'ind_mom_ratio' column
        study_data.loc[:, 'ind_mom_ratio'] = study_data.loc[:, '6m_mom'].divide(study_data.loc[:, 'ind_mom'])

        # JOB: Remove data with missing or unbounded industry momentum ratio
        study_data = study_data[study_data['ind_mom_ratio'].ne(np.inf)]
        study_data.dropna(subset=['ind_mom_ratio'], inplace=True)

        # Standardize ind_mom_ratio
        study_data.loc[:, 'ind_mom_ratio'] = StandardScaler().fit_transform(
            study_data.loc[:, 'ind_mom_ratio'].values.reshape(-1, 1))

    return study_data, study_data_split_index


def create_constituency_matrix(load_from_file=False, index_id='150095', lookup_table='global',
                               folder_path: str = None) -> None:
    """
    Generate constituency matrix for stock market index components

    :param lookup_table: Table to use for constituency lookup
    :param folder_path: Path to data folder
    :param index_id: Index to create constituency matrix for
    :param load_from_file: Flag indicating whether to load constituency information from file

    :return: Tuple containing [0] list of historical constituent gvkeys and [1] relevant date range
    """

    # File name with constituents and corresponding time frames
    file_name = 'data_constituents.csv'
    if folder_path:
        folder = folder_path
    else:
        folder = ''

    db = None
    const_data = None
    parameters = {'index_id': (index_id,)}

    # JOB: In case constituents have to be downloaded from database
    if not load_from_file:
        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')
        print(f'Retrieving index history from {lookup_table} ...')
        if lookup_table == 'global':
            const_data = get_data_table(db, sql_query=True,
                                        query_string="select * "
                                                     "from comp.g_idxcst_his "
                                                     "where gvkeyx in %(index_id)s ",
                                        index_col=['gvkey', 'iid'], table_info=1, params=parameters)
        elif lookup_table == 'north_america':
            const_data = get_data_table(db, sql_query=True,
                                        query_string="select * "
                                                     "from comp.idxcst_his "
                                                     "where gvkeyx in %(index_id)s ",
                                        index_col=['gvkey', 'iid'], table_info=1, params=parameters)

        # Save to file
        const_data.to_csv(os.path.join(ROOT_DIR, folder, 'data_constituents.csv'))

    # JOB: Load table from local file
    else:
        # Load constituency table from file and transform key to string
        const_data = pd.read_csv(os.path.join(ROOT_DIR, folder, file_name), dtype={'gvkey': str})
        # const_data['gvkey'] = const_data['gvkey'].astype('str')

        # Set gvkey and iid as MultiIndex
        const_data.set_index(['gvkey', 'iid'], inplace=True)

    # Convert date columns to datetime format
    for col in ['from', 'thru']:
        const_data.loc[:, col] = pd.to_datetime(const_data[col], format='%Y-%m-%d')
        const_data.loc[:, col] = const_data[col].dt.date

    # Determine period starting date and relevant date range
    index_starting_date = const_data['from'].min()
    relevant_date_range = pd.date_range(index_starting_date, datetime.date.today(), freq='D')

    # Create empty constituency matrix
    constituency_matrix = pd.DataFrame(0, index=relevant_date_range,
                                       columns=const_data.index.drop_duplicates())

    # JOB: Iterate through all company stocks ever listed in index and set adjacency to 0 or 1
    for stock_index in const_data.index:
        if isinstance(const_data.loc[stock_index], pd.Series):
            if pd.isnull(const_data.loc[stock_index, 'thru']):
                constituency_matrix.loc[pd.date_range(start=const_data.loc[stock_index, 'from'],
                                                      end=datetime.date.today()), stock_index] = 1
            else:
                constituency_matrix.loc[pd.date_range(start=const_data.loc[stock_index, 'from'], end=const_data.loc[
                    stock_index, 'thru']), stock_index] = 1
        else:
            for row in const_data.loc[stock_index].iterrows():
                if pd.isnull(row[1]['thru']):
                    constituency_matrix.loc[
                        pd.date_range(start=row[1]['from'], end=datetime.date.today()), stock_index] = 1
                else:
                    constituency_matrix.loc[pd.date_range(start=row[1]['from'], end=row[1]['thru']), stock_index] = 1

    # Save constituency table to file
    constituency_matrix.to_csv(os.path.join(ROOT_DIR, folder, 'constituency_matrix.csv'))

    if not load_from_file:
        db.close()
        print('DB connection closed.')

    return None


def create_gics_matrix(index_id='150095', index_name=None, lookup_table=None, folder_path: str = None,
                       load_from_file=False) -> pd.DataFrame:
    """
    Generate constituency matrix Global Industry Classification Standard (GICS) classification.

    :param index_id: Index to create constituency matrix for
    :param index_name: (Optional) - Index name
    :param lookup_table: (Optional) - Lookup table name
    :param folder_path: (Optional) - Folder path
    :param load_from_file: (Optional) - Tag indicating whether to load matrix from file

    :return:
    """

    lookup_dict = {'Global Dictionary':
                       {'file_path': 'gvkeyx_name_dict.json',
                        'lookup_table': 'global'},
                   'North American Dictionary':
                       {'file_path': 'gvkeyx_name_dict_na.json',
                        'lookup_table': 'north_america'}
                   }

    if load_from_file:
        # JOB: In case existing gics_matrix can be loaded from file
        print(f'Loading GICS matrix for {index_id}: {index_name}')
        timer = Timer()
        timer.start()

        if folder_path is None:
            if index_name is None:
                index_name, lookup_table = lookup_multiple(dict_of_dicts=lookup_dict, index_id=index_id)

            folder_path = os.path.join(ROOT_DIR, 'data',
                                       index_name.lower().replace(' ', '_'))  # Path to index data folder

        # Load from fle
        gic_constituency_matrix = pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'gics_matrix.csv'), index_col=0,
                                              header=0,
                                              parse_dates=True, dtype=str)
        gic_constituency_matrix.index = pd.to_datetime(gic_constituency_matrix.index)

        timer.stop()

    else:
        # JOB: In case gics_matrix has to be created
        if folder_path is None:
            if (index_name is None) or (lookup_table is None):
                index_name, lookup_table = lookup_multiple(dict_of_dicts=lookup_dict, index_id=index_id)

            folder_path = os.path.join(ROOT_DIR, 'data',
                                       index_name.lower().replace(' ', '_'))  # Path to index data folder
        else:
            index_name = str.split(folder_path, '\\')[-1].replace('_', ' ')
            index_id, lookup_table = lookup_multiple(dict_of_dicts=lookup_dict, index_id=index_name,
                                                     reverse_lookup=True, key_to_lower=True)

        folder_exists = check_directory_for_file(index_name=index_name, folder_path=folder_path, create_dir=False)

        if folder_exists:
            print(f'Creating GICS matrix for {index_id}: {index_name.capitalize()}')
            timer = Timer().start()

        else:
            print(
                f'Directory for index {index_id} ({index_name.capitalize()}) does not exist. \nPlease either download the '
                f'necessary data or choose different index ID.')
            raise LookupError('Value not found.')

        # JOB: Get all historic index constituents
        gvkey_list, _ = get_all_constituents(
            constituency_matrix=pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'constituency_matrix.csv'),
                                            index_col=0,
                                            header=[0, 1],
                                            parse_dates=True))

        parameters = {'index_ids': tuple(gvkey_list)}

        # JOB: Download GICS table
        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')
        print(f'Retrieving GICS history from {lookup_table} ...')
        if lookup_table == 'global':
            const_data = get_data_table(db, sql_query=True,
                                        query_string="select * "
                                                     "from comp.g_co_hgic "
                                                     "where gvkey in %(index_ids)s ",
                                        index_col=['gvkey'], table_info=1, params=parameters)
        elif lookup_table == 'north_america':
            const_data = get_data_table(db, sql_query=True,
                                        query_string="select * "
                                                     "from comp.co_hgic "
                                                     "where gvkey in %(index_ids)s ",
                                        index_col=['gvkey'], table_info=1, params=parameters)
        else:
            raise LookupError('Value not found.')

        # Convert date columns to datetime format
        for col in ['indfrom', 'indthru']:
            const_data.loc[:, col] = pd.to_datetime(const_data[col], format='%Y-%m-%d')
            const_data.loc[:, col] = const_data[col].dt.date

        # Determine period starting date and relevant date range
        index_starting_date = const_data['indfrom'].min()
        relevant_date_range = pd.date_range(index_starting_date, datetime.date.today(), freq='D')

        # Create empty constituency matrix
        gic_constituency_matrix = pd.DataFrame(None, index=relevant_date_range,
                                               columns=const_data.index.drop_duplicates())

        # JOB: Iterate through all company stocks ever listed in index and set adjacency to 0 or 1
        for stock_index in const_data.index:
            if isinstance(const_data.loc[stock_index], pd.Series):
                if pd.isnull(const_data.loc[stock_index, 'indthru']):
                    gic_constituency_matrix.loc[
                        pd.date_range(start=const_data.loc[stock_index, 'indfrom'],
                                      end=datetime.date.today()), stock_index] = const_data.loc[stock_index, 'gsubind']
                else:
                    gic_constituency_matrix.loc[
                        pd.date_range(start=const_data.loc[stock_index, 'indfrom'], end=const_data.loc[
                            stock_index, 'indthru']), stock_index] = const_data.loc[stock_index, 'gsubind']
            else:
                for row in const_data.loc[stock_index].iterrows():
                    if pd.isnull(row[1]['indthru']):
                        gic_constituency_matrix.loc[
                            pd.date_range(start=row[1]['indfrom'], end=datetime.date.today()), stock_index] = row[1][
                            'gsubind']
                    else:
                        gic_constituency_matrix.loc[
                            pd.date_range(start=row[1]['indfrom'], end=row[1]['indthru']), stock_index] = row[1][
                            'gsubind']

        # Save constituency table to file
        gic_constituency_matrix.to_csv(os.path.join(ROOT_DIR, folder_path, 'gics_matrix.csv'))

        db.close()
        print('DB connection closed.')

        timer.stop()

    # JOB: Stack columns onto index
    gic_constituency_matrix = gic_constituency_matrix.stack()
    gic_constituency_matrix.index.set_names(['datadate', 'gvkey'], inplace=True)
    gic_constituency_matrix.name = 'gsubind'

    return gic_constituency_matrix


def calculate_daily_return(data: pd.DataFrame, save_to_file=False, folder_path=None):
    """
    Add return index and daily return columns to Data Frame

    :param folder_path: Path to index data folder
    :param save_to_file: Flag indicating whether to save appended Data Frame to original file
    :param data: Original Data Frame
    :return: Data Frame with added columns
    """
    data = data.reset_index(inplace=False).set_index(keys=['datadate', 'gvkey', 'iid'], inplace=False)

    # JOB: Calculate Return Index Column
    data.loc[:, 'return_index'] = (data['prccd'] / data['ajexdi']) * data['trfd']

    # JOB: Calculate Daily Return
    data.loc[:, 'daily_return'] = data.groupby(level=['gvkey', 'iid'])['return_index'].apply(
        lambda x: x.pct_change(periods=1))

    # JOB: Filter out observations with zero trading volume and zero daily return (public holidays)
    data = data[~(data['cshtrd'].isna() & (data['daily_return'] == 0))]

    # JOB: Recalculate Daily Return
    data.loc[:, 'daily_return'] = data.groupby(level=['gvkey', 'iid'])['return_index'].apply(
        lambda x: x.pct_change(periods=1))

    if save_to_file:
        # Reset index
        data.reset_index(inplace=True)

        # Save to file
        data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents_test.csv'))

    return data


def get_index_constituents(constituency_matrix: pd.DataFrame, date: datetime.date, folder_path: str,
                           print_constituents=False) -> pd.Index:
    """
    Return company name list (pd.Index) of index constituents for given date

    :param print_constituents: Flag indicating whether to print out index constituent names
    :param folder_path: Path to file directory
    :param constituency_matrix: Constituency table providing constituency information
    :param date: Date for which to return constituency list

    :return: Index of company identifiers for given date
    """

    try:
        lookup_dict = pd.read_json(os.path.join(ROOT_DIR, folder_path, 'gvkey_name_dict.json'),
                                   typ='series').to_dict().get(
            'conm')
    except ValueError as ve:
        lookup_dict = None
        print_constituents = False

    # print(lookup_dict)
    print(
        f'Number of constituents (index constituency matrix): '
        f'{len(constituency_matrix.loc[date].loc[lambda x: x == 1])}')
    # print(
    #     f"List of constituents: {constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values('gvkey').tolist()}")

    if print_constituents:
        constituent_list_names = [lookup_dict.get(key) for key in
                                  constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values(
                                      'gvkey').tolist()]
        for company_name in constituent_list_names:
            print(company_name)

    return constituency_matrix.loc[date].loc[lambda x: x == 1].index


def get_all_constituents(constituency_matrix: pd.DataFrame) -> tuple:
    """
    Return all historical constituents' gvkeys and full date range from constituency matrix

    :param constituency_matrix: Matrix to extract constituents and time frame from
    :return: Tuple containing [0] list of historical constituent gvkeys and [1] relevant date range
    """

    # Query complete list of historic index constituents' gvkeys
    constituent_list = constituency_matrix.columns.get_level_values('gvkey').drop_duplicates().to_list()

    # Query relevant date range
    index_starting_date = constituency_matrix.index.sort_values()[0]
    relevant_date_range = pd.date_range(index_starting_date, datetime.date.today(), freq='D')

    return constituent_list, relevant_date_range


def get_data_table(db: wrds.Connection, sql_query=False, query_string='', library=None, table=None, columns=None,
                   obs=-1,
                   index_col=None, date_cols=None, recent=False, n_recent=100, params=None,
                   table_info=1) -> pd.DataFrame:
    """
    Get pandas DataFrame containing the queries data from the specified library and table.

    :param db: wrds database connection
    :param sql_query: SQL query flag
    :param query_string: SQL query string
    :param library: Library to use
    :param table: Table withing library
    :param columns: Columns to include in the DataFrame
    :param obs: Number of observations
    :param index_col: Index of index columns
    :param date_cols: Index of data columns
    :param recent: Flag indicating whether to only return recent data
    :param n_recent: Number of recent data rows to return
    :param table_info: Integer indicating info verbosity
    :param params: Addition SQL query parameter
    :return: DataFrame containing the queried table
    """

    print('Processing query ...')

    # JOB: Extract library and table name from SQL query string
    if sql_query and query_string:
        lib_tab_string = re.search('(?:from|FROM)\s(\S*)', query_string).group(1)
        library, table = lib_tab_string.split('.')

    # JOB: Print information about the queried table
    if table_info == 1:
        print('Approximate number of rows in %s.%s: %d' % (library, table, db.get_row_count(library, table)))
    elif table_info == 2:
        print('Queried table information for %s - %s: ' % (library, table))
        print(db.describe_table(library, table))

    if not sql_query:
        # Standard non-SQL query
        data_table = db.get_table(library=library, table=table, columns=columns, obs=obs, index_col=index_col,
                                  date_cols=date_cols)

    else:
        # Query via SQL string
        data_table = db.raw_sql(query_string, index_col=index_col, date_cols=date_cols, params=params)

    if recent:
        # In case only n_recent dates are queried

        # Create array of last n date indices (deduplicated)
        last_n_indices = data_table.set_index(keys='datadate').index.drop_duplicates().sort_values(
            ascending=True).values[
                         -n_recent:]

        data_table.set_index(keys='datadate', inplace=True)

        # Query last n dates from DataFrame
        data_table = data_table.loc[last_n_indices, :]

    return data_table


def list_tables(db, library, show_n_rows=False):
    """
    List all tables in a given library with optional row count information
    :param db: Database connection
    :param library: Queries library
    :param show_n_rows: Show number of rows
    :return:
    """
    for table in db.list_tables(library):
        if show_n_rows:
            print(table + ': %d rows' % db.get_row_count(library, table))
        else:
            print(table)


def generate_index_lookup_dict() -> None:
    """
    Generate dictionary mapping GVKEYX (index identifier) to index name

    :return:
    """

    gvkeyx_lookup = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'Compustat_Global_Indexes.csv'), index_col='GVKEYX')
    gvkeyx_lookup.index = gvkeyx_lookup.index.astype(str)
    gvkeyx_lookup = gvkeyx_lookup['index_name'].to_dict()

    with open(os.path.join(ROOT_DIR, 'data', 'gvkeyx_name_dict.json'), 'w') as fp:
        json.dump(gvkeyx_lookup, fp)


def generate_company_lookup_dict(folder_path: str, data: pd.DataFrame) -> None:
    """
    Generate dictionary mapping GVKEY (stock identifier) to stock name

    :return:
    """

    if 'conm' not in data.columns:
        data.rename(columns={'conml': 'conm'}, inplace=True)

    # JOB: Create company name dictionary and save to file
    company_name_lookup = data.reset_index()
    company_name_lookup = company_name_lookup.loc[
        ~company_name_lookup.duplicated(subset='gvkey', keep='first'), ['gvkey', 'conm']].set_index('gvkey')

    # JOB: Save dict to json file
    company_name_lookup.to_json(os.path.join(ROOT_DIR, folder_path, 'gvkey_name_dict.json'))


def generate_gics_sector(data: pd.DataFrame):
    """
    Append additional column with 2-digit GICS code

    :param data: Original Data Frame
    :return:
    """

    print('Generating gics_sector column ...')
    timer = Timer().start()
    data.loc[:, 'gics_sector'] = data['gsubind'].apply(lambda x: str(x)[:2] if not pd.isna(x) else '')
    timer.stop()


def append_columns(data: pd.DataFrame, folder_path: str, file_name: str, column_list: list):
    """

    :param column_list:
    :param data:
    :param folder_path:
    :param file_name:
    :return:
    """

    # TODO: Make merge work
    index_col_names = data.index.names
    # Load DataFrame with new columns
    new_col_df = pd.read_csv(os.path.join(ROOT_DIR, folder_path, file_name), index_col=['datadate', 'gvkey', 'iid'],
                             header=0,
                             parse_dates=True, infer_datetime_format=True,
                             dtype={'gvkey': str})

    new_col_df = new_col_df.loc[:, column_list]

    # JOB: Merge data with new columns

    timer = Timer().start()
    if None in index_col_names:
        data.reset_index(inplace=True)
        print('Resetting existing index.')
        data.drop(columns='index', inplace=True)
    else:
        data.reset_index(inplace=True, drop=True)
        print('Resetting non-existent index.')

    print('Merging full data set with new columns ...')
    data.set_index(['datadate', 'gvkey', 'iid'], inplace=True)
    data = data.merge(new_col_df, how='left', left_index=True, right_index=True)
    data.reset_index(inplace=True)
    timer.stop()
    print(data.head())
    print(data.tail())
    print(data.sample(40))

    # Save to file
    print('Saving appended DataFrame to file ...')
    timer = Timer().start()
    data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents_test.csv'))
    timer.stop()

    if index_col_names is not None:
        data.set_index(index_col_names, inplace=True)
    else:
        data.reset_index(inplace=True, drop=True)


def add_constituency_col(data_orig: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    """
    Append constituency column to DataFrame

    :param data_orig: Original DataFrame to add column to
    :param folder_path: Path to index data folder
    :return: Appended DataFrame
    """
    print('\nAdding constituency column ...')
    timer = Timer().start()
    constituency_matrix = pd.read_csv(os.path.join(ROOT_DIR, folder_path, 'constituency_matrix.csv'),
                                      index_col=0,
                                      header=[0, 1],
                                      parse_dates=True).stack([0, 1]).rename('in_index')

    constituency_matrix.index.rename(['datadate', 'gvkey', 'iid'], inplace=True)

    data = data_orig.copy()
    del data_orig
    saved_index = data.index.names
    data.reset_index(inplace=True)

    if None in saved_index:
        data.drop(columns='index', inplace=True)

    data.set_index(['datadate', 'gvkey', 'iid'], inplace=True)

    data = data.merge(constituency_matrix, how='left', left_index=True, right_index=True)

    data.reset_index(inplace=True)
    try:
        if None not in saved_index:
            data.set_index(saved_index, inplace=True)
    except IndexError as ie:
        pass
    print('Done')
    timer.stop()

    return data


def quickload_full_data(dax=False):
    path = r'C:\Users\svenk\PycharmProjects\Financial\data\dow_jones_stoxx_600_price_index'
    if dax:
        path = r'C:\Users\svenk\PycharmProjects\Financial\data\deutscher_aktienindex_(dax)_index'

    data = pd.read_csv(os.path.join(path, r'index_data_constituents.csv'),
                       index_col='datadate', header=0, parse_dates=True, infer_datetime_format=True,
                       dtype={'gvkey': str, 'gsubind': str, 'gics_sector': str})
    data = add_constituency_col(data, path)
    data = data.loc[data['in_index'] == 1, :]
    data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')], inplace=True)
    gics_map = pd.read_json(os.path.join('data', 'gics_code_dict.json'), orient='records', typ='series').rename(
        'gics_sec')
    gics_map = {str(key): val for key, val in gics_map.to_dict().items()}
    data['gics_sec'] = data.loc[:, 'gics_sector'].replace(gics_map, inplace=True)

    return data


def to_actual_index(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data:
    :return:
    """
    data_new = data.loc[data['in_index'] == 1, :]
    del data
    return data_new
