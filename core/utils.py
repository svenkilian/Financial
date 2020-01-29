"""
This utilities module implements helper functions for displaying data frames and plotting data.
"""
import itertools
import webbrowser

import config
from config import *
import csv
import datetime
import datetime as dt
import glob
import json
import sys
import time
from collections import deque
from pydoc import locate

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate


def plot_data(data: pd.DataFrame, columns: list = None, index_name: str = None, title='', col_multi_index=False,
              label_mapping=None, export_as_png=False) -> None:
    """
    Plot data

    :param data: DataFrame containing the data to plot
    :param columns: Columns to plot
    :param index_name: Name of the row index
    :param title: Title of the plot
    :param col_multi_index: Flag indicating whether MultiIndex is used in columns
    :param label_mapping: Dict mapping label codes to descriptive text
    :param export_as_png: Flag indicating whether to export plot to png
    :return: None

    Usage::
    Plotting all rows and columns:
            >>> plot_data(data.loc[:], columns=data.columns, index_name=None, title='Indices', col_multi_index=False,
            >>> label_mapping=indices, export_as_png=True)
    """

    # Initialize date locators and formatters
    # if index_name is None:
    #     index_name = list('datadate')
    years = mdates.YearLocator(5)
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')
    months_fmt = mdates.DateFormatter('%m')

    # Initialize plot
    fig, ax = plt.subplots()

    # Reset index and change to index_name if provided
    if index_name:
        data = data.reset_index().set_index(index_name)
        if len(list(index_name)) == 1:
            data.index = pd.to_datetime(data.index)
            print(data.index)

    # Plot columns
    ax.plot(data.index, data[columns], '-', linewidth=1.2, markersize=2)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    # ax.xaxis.set_minor_formatter(months_fmt)

    # Round to nearest years
    # datemin = np.datetime64(data.index[0], 'Y')
    # datemax = np.datetime64(data.index[-1], 'Y') + np.timedelta64(1, 'Y')
    # ax.set_xlim(datemin, datemax)

    # Style plot
    ax.set(title=title, xlabel='Year', ylabel='Value')
    ax.grid(True)
    if col_multi_index:
        if label_mapping:
            ax.legend(label_mapping.values())
        else:
            ax.legend(columns.get_level_values(1))
    else:
        ax.legend(columns)

    if export_as_png:
        plt.savefig('plot.png', dpi=1200, facecolor='w', bbox_inches='tight')

    plt.show()


def pretty_print_table(df: pd.DataFrame, headers='keys', tablefmt='fancy_grid'):
    """
    Print out DataFrame in tabular format.

    :type headers: list or string
    :param df: DataFrame to print
    :param headers: Table headers
    :param tablefmt: Table style
    :return:

    Usage::
            >>> pretty_print_table(db, headers=['col_1', 'col_2'])
    """
    print(tabulate(df, headers, tablefmt=tablefmt, showindex=True, floatfmt='.4f'))


def plot_train_val(history, metrics: list, store_png=False, folder_path='') -> None:
    """
    Plot training and validation metrics over epochs.

    :param history: Training history
    :param metrics: List of metrics to plot
    :type metrics: list of strings
    :return: None
    """

    # Plot history for all metrics
    epoch_axis = range(1, len(history.history['loss']) + 1)
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.plot(epoch_axis, history.history[metric])
        ax.plot(epoch_axis, history.history['val_' + metric])
    ax.set_title(metrics[0].capitalize())
    ax.set_ylabel(metrics[0].capitalize())
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    if store_png:
        fig.savefig(os.path.join(ROOT_DIR, folder_path, 'metrics.png'), dpi=600)

    # Plot history for loss
    fig, ax = plt.subplots()
    ax.plot(epoch_axis, history.history['loss'])
    ax.plot(epoch_axis, history.history['val_loss'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(['Training', 'Validation'], loc='upper left')

    plt.show()
    if store_png:
        fig.savefig(os.path.join(ROOT_DIR, folder_path, 'loss.png'), dpi=600)


def plot_results(predicted_data, true_data):
    """
    Plot predictions vs. true labels.

    :param predicted_data:
    :param true_data:
    :return:
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def df_to_html(df: pd.DataFrame, title='', file_name='study_period_data', open_window=True):
    """
    Create and save HTML table

    :param df: DataFrame
    :param title: Title of HTML page
    :param file_name: File name to save HTML page to
    :param open_window: Open created file in new browser window
    :return:
    """
    table = df.to_html(classes='blueTable')
    html_doc = f"""
    <html>
        <head><title>{title}</title></head>
        <link rel="stylesheet" type="text/css" media="screen" href="tbl_style.css" />
        <body>
            {table}
        </body>
    </html>
    """
    with open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), 'w') as f:
        f.write(html_doc)

    if open_window:
        webbrowser.open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), new=2)


def load_json_to_df(file_path: str) -> pd.DataFrame:
    """
    Load data from json file specified in path and return as pandas data frame.

    :param file_path: Path to json file
    :return: Data frame containing data from json file
    """

    # Specify path to saved data
    path_to_data = os.path.join(ROOT_DIR, file_path)

    # Load json data as dict
    with open(path_to_data, 'r') as file:
        data = json.load(file)

    # Make DataFrame from json
    data_frame = pd.DataFrame(data)

    return data_frame


def get_most_recent_file(directory: str) -> str:
    """
    Retrieve name of most recently changed file in given directory.

    :param directory: Directory to search for most recently edited file
    :return: Name of most recently edited file
    """
    all_files = glob.glob(directory + '/*')
    most_recent_file = max(all_files, key=os.path.getctime)

    # print(all_files)
    # print(most_recent_file)

    return most_recent_file


def lookup_multiple(dict_of_dicts: dict = None, index_id: str = '', reverse_lookup=False, key_to_lower=False) -> (
        str, str):
    """
    Retrieve index name and lookup table name for given index id and collection of lookup dicts.

    The nested dictionary "dict_of_dicts" should have the following structure:

    {'Global Dictionary':
             {'file_path': 'gvkeyx_name_dict.json',
              'lookup_table': 'global'},
    'North American Dictionary':
             {'file_path': 'gvkeyx_name_dict_na.json',
              'lookup_table': 'north_america'}}

    :param key_to_lower:
    :param reverse_lookup:
    :param dict_of_dicts: Nested dictionary with lookup dicts
    :param index_id: Index ID
    :return: Tuple of [0] index_name, [1] lookup_table
    """

    if dict_of_dicts is None:
        dict_of_dicts = {'Global Dictionary':
                             {'file_path': 'gvkeyx_name_dict.json',
                              'lookup_table': 'global'},
                         'North American Dictionary':
                             {'file_path': 'gvkeyx_name_dict_na.json',
                              'lookup_table': 'north_america'}}

    # Load index name dict and get index name
    for key, value in dict_of_dicts.items():
        print(f'Trying lookup in {key} ...')
        lookup_dict = json.load(open(os.path.join(ROOT_DIR, 'data', value.get('file_path')), 'r'))

        if reverse_lookup:
            lookup_dict = {v: k for k, v in lookup_dict.items()}
        if key_to_lower:
            lookup_dict = {k.lower(): v for k, v in lookup_dict.items()}

        lookup_table = value.get('lookup_table')
        index_name = lookup_dict.get(index_id)  # Check whether index id is in global dict
        if index_name is not None:
            print(f'Found index ID in {key}: {index_name}\n')
            break
        else:
            # In case index id is not in dict, search in remaining dicts
            print(f'Index identifier not in {key}.')
    else:
        print('Index ID not found in any dictionary.')
        raise LookupError('Index ID not known.')

    return index_name, lookup_table


def add_to_json(dict_entry: dict, file_path: str):
    """
    Add entry to json file

    :param dict_entry: Entry to add (dict)
    :param file_path: Path to json file
    :return: Appended json
    """

    # Specify path to saved data
    path_to_data = os.path.join(ROOT_DIR, file_path)

    if not os.path.exists(path_to_data):
        print(f'Creating json file: {path_to_data}')
        with open(path_to_data, 'w') as f:
            json.dump({}, f)

    with open(path_to_data) as f:
        data = json.load(f)

    id = int(datetime.datetime.today().timestamp())
    data.update({id: dict_entry})

    with open(path_to_data, 'w') as f:
        json.dump(data, f)

    return data


def check_directory_for_file(index_name: str = None, index_id=None, folder_path: str = '', force_download: bool = False,
                             create_dir=True, print_status=True) -> bool:
    """
    Check whether folder path already exists.

    :param print_status: Whether to print status message
    :param index_id: Index ID
    :param index_name: Name of index
    :param folder_path: Folder path to check
    :param force_download: Flag indicating whether to force download into existing folder
    :param create_dir: Flag indicating whether to create new directory if path does not exist
    :return: Boolean indicating whether to load from file (directory exists)
    """

    if (index_name is None) and (index_id is not None):
        index_name, _ = lookup_multiple(index_id=index_id)

    if os.path.exists(os.path.join(ROOT_DIR, folder_path)):
        if force_download:
            load_from_file = False
            print(f'Downloading data from {index_name} into existing folder: {folder_path}\n')
        else:
            load_from_file = True
            if print_status:
                print(f'Loading data from {index_name} from folder: {folder_path}\n')
    else:
        if create_dir:
            print(f'Creating folder for {index_name} %s')
            os.mkdir(os.path.join(ROOT_DIR, folder_path))
        load_from_file = False

    return load_from_file


def to_combinations(list_of_objs: list):
    """
    Create array of all combinations of a list of length n of the shape (1, 2, ..., n)
    :param list_of_objs:
    :return:
    """
    combinations = []
    list_of_lists = []

    for i in range(2, len(list_of_objs)):
        combinations.extend(itertools.combinations(range(len(list_of_objs)), i))

    for comb in combinations:
        item_list = []
        for item in comb:
            item_list.append(list_of_objs[item])
        list_of_lists.append(item_list)

    return list_of_lists


def get_run_number():
    """
    Get number ('ID') of current run from log file.

    :return:
    """
    try:
        # config.run_id = CSVReader(os.path.join(ROOT_DIR, 'data', 'training_log.csv')).get_last_row_value('ID',
        #                                                                                                  cast_type='int') + 1

        with open(os.path.join(ROOT_DIR, 'data', 'training_log.json'), 'r') as f:
            data = json.load(f)
            last_item = deque(data.items(), maxlen=1).pop()[1]
            if last_item.get('ID'):
                config.run_id = int(last_item.get('ID') + 1)
            else:
                print('Last entry has no ID. Looking for largest ID instead.')
                max_key = max([val.get('ID', 0) for val in data.values() if val.get('ID') is not None])
                config.run_id = int(max_key + 1)
    except FileNotFoundError:
        config.run_id = 1

    return config.run_id


def apply_batch_directory(directory_path, function=None, **func_args):
    """
    Apply given function to all subdirectories in a given parent directory.

    :param directory_path: Path of the parent directory
    :param function: Function to apply to direct subdirectories
    :param func_args: Optional function arguments
    :return:
    """
    index_directories = [name for name in os.listdir(os.path.join(ROOT_DIR, directory_path))
                         if os.path.isdir(os.path.join(ROOT_DIR, directory_path, name))]

    for index_dir in index_directories:
        full_path = os.path.join(ROOT_DIR, directory_path, index_dir)
        function(folder_path=full_path, **func_args)


def check_data_conformity(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
                          test_data_index: pd.MultiIndex):
    """
    Check for conformity of given training and test sets (including test set index)

    :param x_train: Training data input
    :param y_train: Training data labels
    :param x_test: Test data input
    :param y_test: Test data labels
    :param test_data_index: Test data index
    """

    # JOB: Perform data size conformity checks
    print('\nChecking for training data size conformity: %s' % (len(x_train) == len(y_train)))
    print('Checking for test data size conformity: %s' % (len(x_test) == len(y_test)))
    print('Checking for test data index size conformity: %s \n' % (len(y_test) == len(test_data_index)))

    if (len(x_train) != len(y_train)) or (len(x_test) != len(y_test)):
        print(f'{Fore.RED}{Back.YELLOW}Training data lengths do not conform!{Style.RESET_ALL}')
        raise AssertionError('Training data lengths do not conform.')

    if len(y_test) != len(test_data_index):
        print(f'{Fore.RED}{Back.YELLOW}Test data lengths do not conform!{Style.RESET_ALL}')
        raise AssertionError('Test data index length is not conforming.')

    # JOB: Determine target label distribution in train and test sets and check for plausibility
    target_mean_train = np.mean(y_train)
    assert target_mean_train <= 1
    target_mean_test = np.mean(y_test)
    assert (y_test <= 1).all()
    assert target_mean_test <= 1

    return target_mean_train, target_mean_test


def print_study_period_ranges(full_data: pd.DataFrame, study_period_ranges: dict) -> None:
    """

    :param study_period_ranges: Dict containing the study period index and ranges
    :param full_data:
    """

    full_date_range = full_data['datadate'].unique()
    print('\nStudy period date index ranges:')
    for study_period_no in list(sorted(study_period_ranges.keys())):
        date_index_tuple = study_period_ranges.get(study_period_no)
        date_tuple = tuple(
            pd.to_datetime(date_full).date().isoformat() for date_full in full_date_range[list(date_index_tuple)])
        print(
            f'Period {Style.BRIGHT}{Fore.YELLOW}{study_period_no}{Style.RESET_ALL}: {date_index_tuple} -> {date_tuple}')


def get_model_parent_type(configs: dict = None, model_type: str = None):
    """
    Get parent model type for classifier

    :param configs: Configurations dict
    :param model_type: Model type to get parent type for
    """
    if configs is None:
        configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))

    model_type_reverse_dict = {child_type: parent_type for parent_type in configs['model_hierarchy'].keys() for
                               child_type in
                               configs['model_hierarchy'][parent_type]}

    # Return parent model type from dict
    return model_type_reverse_dict.get(model_type)


def annualize_metric(metric: float, holding_periods: int = 1) -> float:
    """
    Annualize metric of arbitrary periodicity

    :param metric: Metric to analyze
    :param holding_periods:
    :return: Annualized metric
    """

    days_per_year = 360
    trans_ratio = days_per_year / holding_periods

    return (1 + metric) ** trans_ratio - 1


def to_monthly(metric: float, holding_periods: int = 1):
    """
    Transform metric of arbitrary periodicity to monthly

    :param metric:
    :param holding_periods:
    :return:
    """

    trans_ratio = 30 / holding_periods

    return (1 + metric) ** trans_ratio - 1


def get_study_period_ranges(data_length: int, test_period_length: int, study_period_length: int, index_name: str,
                            reverse=True,
                            verbose=1) -> dict:
    """
    Get study period ranges, going backwards in time, as a dict

    :param data_length: Number of total dates in index data
    :param test_period_length: Length of test period
    :param study_period_length: Length of study period
    :param index_name: Index name
    :param reverse:
    :param verbose:
    :return: Dict of study period ranges
    """

    n_full_periods = (data_length - (study_period_length - test_period_length)) // test_period_length
    remaining_days = (data_length - (study_period_length - test_period_length)) % test_period_length

    if verbose == 1:
        print(f'Data set contains {data_length} individual dates.')
        print(f'Available index history for {index_name} allows for {n_full_periods} overlapping study periods.')
        print(f'{remaining_days} days are excluded from study data.')

    study_period_ranges = {}
    for period in range(n_full_periods):
        study_period_ranges[period + 1] = (
            - (period * test_period_length + study_period_length), -(period * test_period_length + 1))

    if reverse:
        # Reverse dict such that most recent period has largest index
        dict_len = len(study_period_ranges.keys())
        study_period_ranges = {dict_len - key + 1: value for key, value in study_period_ranges.items()}

    return study_period_ranges


def get_index_name(index_id: str, lookup_dict: dict = None) -> (str, str):
    """
    Get index name and corresponding lookup table based on index ID

    :param index_id: Index ID to look up name and lookup table for
    :param lookup_dict: Optional custom lookup dict. Defaults to standard gvkeyx lookup dict
    :return:
    """

    if lookup_dict is None:
        lookup_dict = {'Global Dictionary':
                           {'file_path': 'gvkeyx_name_dict.json',
                            'lookup_table': 'global'},
                       'North American Dictionary':
                           {'file_path': 'gvkeyx_name_dict_na.json',
                            'lookup_table': 'north_america'}}

    index_name, lookup_table = lookup_multiple(lookup_dict, index_id=index_id)

    return index_name, lookup_table


def calc_sharpe(return_series, annualize=True):
    """
    Calculate Sharpe Ratio for series of returns

    :param return_series:
    :param annualize:
    :return:
    """

    days_per_year = 365
    excess_returns = calc_excess_returns(return_series)
    std = np.std(excess_returns)
    res = np.divide(excess_returns.mean(), std)

    if annualize:
        res = res * np.sqrt(days_per_year)

    return res


def calc_sortino(return_series, annualize=True):
    """
    Calculate Sortino Ratio for series of returns

    :param return_series:
    :param annualize:
    :return:
    """

    days_per_year = 365
    excess_returns = calc_excess_returns(return_series)
    semi_std = np.std(excess_returns[excess_returns < 0])
    res = np.divide(excess_returns.mean(), semi_std)

    if annualize:
        res = res * np.sqrt(days_per_year)

    return res


def deannualize(return_series, n_periods=365):
    """
    Convert return in annual terms to a daily basis

    :param return_series: Return series to deannualize
    :param n_periods: Target basis (365 for daily)
    :return:
    """
    return np.power(1 + return_series, 1.0 / n_periods) - 1


def calc_excess_returns(return_series: pd.DataFrame, rf_rate_series: pd.DataFrame = None):
    """
    Calculate daily excess returns for a given return series and a series of the risk-free rate (annual values)

    :param return_series: Return series to calculate excess returns for
    :param rf_rate_series: Series with risk-free rate (annual percentage)
    :return:
    """
    if rf_rate_series is None:
        rf_rate_series = (pd.read_csv(os.path.join(ROOT_DIR, 'data', 'rf_rate_germany.csv'), parse_dates=True,
                                      index_col=0) / 100).resample('D').ffill()
    combined = pd.merge(return_series, deannualize(rf_rate_series), how='left', left_index=True, right_index=True)
    excess_returns = combined['daily_return'].subtract(combined['rf_rate_pct'])

    return excess_returns


def countdown(seconds: int):
    while seconds >= 0:
        sys.stdout.flush()
        mins, secs = divmod(seconds, 60)
        time_remaining = 'Trying again in {:02d}:{:02d}'.format(mins, secs)
        sys.stdout.write(f'\r{time_remaining}')
        time.sleep(1)
        seconds -= 1
    sys.stdout.write('\r ')
    sys.stdout.flush()


def get_run_id(logfile_name: str):
    file_path = os.path.join(ROOT_DIR, 'data', logfile_name, '.csv')
    reader = CSVReader(file_path)


class ProgressBar:
    """
    Class implementing a progress bar
    """

    def __init__(self, total):
        self.start_time = time.time()
        self.total = total
        self.lapsed_time = 0.0
        self.iteration = 0

    def print_progress(self, prefix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar

        :param prefix: Prefix string (Str)
        :param decimals: Positive number of decimals in percent complete (Int)
        :param bar_length: Character length of bar (Int)
        """

        self.iteration += 1
        self.lapsed_time = time.time() - self.start_time
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (self.iteration / float(self.total)))
        filled_length = int(round(bar_length * self.iteration / float(self.total)))
        bar = u'\u258B' * filled_length + '-' * (bar_length - filled_length)
        pending_time = (self.lapsed_time / self.iteration) * (self.total - self.iteration)
        minutes, seconds = divmod(pending_time, 60)
        suffix = f'{int(minutes)} mins, {str_format.format(seconds)} secs remaining'
        sys.stdout.write(
            '\r%s |%s| %s%s - Processing stock %d of %d - %s' % (
                prefix, bar, percents, '%', self.iteration, self.total, suffix)),

        if self.iteration == self.total:
            minutes, seconds = divmod(time.time() - self.start_time, 60)
            sys.stdout.write(f'\nTime taken: {int(minutes)} mins, {str_format.format(seconds)} secs')
            sys.stdout.write('\n')

        sys.stdout.flush()


class Timer:
    """
    Timer class for timing operations.
    """

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()
        return self

    def stop(self):
        end_dt = dt.datetime.now()
        print(f'{Style.DIM}{Fore.LIGHTBLUE_EX}Time taken: %s{Style.RESET_ALL}' % (end_dt - self.start_dt))


class CSVWriter:
    """
    Writer class for handling the logging of training runs into a .csv file.
    """

    def __init__(self, output_path: str, field_names: list):
        self.output_path = output_path
        self.field_names = field_names
        self.delimiter = ';'
        self.lineterminator = '\n'
        self.access_delay = 5

        if not os.path.exists(os.path.join(ROOT_DIR, output_path)):
            with open(self.output_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.field_names, delimiter=self.delimiter,
                                        lineterminator=self.lineterminator)
                writer.writeheader()

    def add_line(self, record: dict):
        """
        Add data line (dict format) to .csv file.

        :param record: Dictionary of records to add to file
        :return:
        """

        try:
            with open(self.output_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.field_names, delimiter=self.delimiter,
                                        lineterminator=self.lineterminator)

                # Append current record
                writer.writerow(record)
        except PermissionError as pe:
            print(f'{Style.BRIGHT}{Fore.RED}File access to {pe.filename} denied. \n'
                  f'Make sure the file is closed to enable logging.')

            # Pause execution for specified time
            countdown(self.access_delay)
            print(f'{Style.RESET_ALL}')
            # Try again
            print('Trying again ...')
            self.add_line(record)


class CSVReader:
    """
    Reader class for handling the logging of training runs into a .csv file.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.delimiter = ';'
        self.lineterminator = '\n'
        self.access_delay = 5

    @property
    def field_names(self):
        try:
            with open(self.file_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=self.delimiter,
                                        lineterminator=self.lineterminator)
                return reader.fieldnames

        except BaseException as be:
            print(be.args)

    def get_last_row(self):
        try:
            with open(self.file_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=self.delimiter,
                                        lineterminator=self.lineterminator)

                d = deque(reader, maxlen=1)
                return d.pop()

        except BaseException as be:
            print(be.args)

    def get_last_row_value(self, key, cast_type=None):
        if self.path_exists():
            try:
                with open(self.file_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter=self.delimiter,
                                            lineterminator=self.lineterminator)

                    d = deque(reader, maxlen=1)
                    ret_val = d.pop().get(key, None)
                    if cast_type:
                        ret_val = locate(cast_type)(ret_val)
                    return ret_val

            except BaseException as be:
                print(be.args)

        else:
            # print(f'File path {self.file_path} does not exist.')
            raise FileNotFoundError(f'{Style.BRIGHT}{Fore.RED}File does not exist.{Style.RESET_ALL}')

    def path_exists(self):
        return os.path.exists(os.path.join(ROOT_DIR, self.file_path))
