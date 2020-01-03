"""
This utilities module implements helper functions for displaying data frames and plotting data.
"""
import datetime as dt
import glob
import json
import os

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import ROOT_DIR
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters
from pygments.lexers import configs
from tabulate import tabulate
from colorama import Fore, Back, Style
import logging
import csv
import io

# Update matplotlib setting
plt.rcParams.update({'legend.fontsize': 8,
                     'legend.loc': 'best',
                     'legend.markerscale': 2.5,
                     'legend.frameon': True,
                     'legend.fancybox': True,
                     'legend.shadow': True,
                     'legend.facecolor': 'w',
                     'legend.edgecolor': 'black',
                     'legend.framealpha': 1})

# matplotlib.style.use('fivethirtyeight')
matplotlib.style.use('ggplot')
register_matplotlib_converters()


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


def pretty_print(df: pd.DataFrame, headers='keys', tablefmt='fancy_grid'):
    """
    Print out DataFrame in tabular format.

    :type headers: list or string
    :param df: DataFrame to print
    :param headers: Table headers
    :param tablefmt: Table style
    :return:

    Usage::
            >>> pretty_print(db, headers=['col_1', 'col_2'])
    """
    print(tabulate(df, headers, tablefmt=tablefmt, showindex=True))


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


def lookup_multiple(dict_of_dicts: dict = None, index_id: str = '', reverse_lookup=False, key_to_lower=False):
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


def check_directory_for_file(index_name: str = '', folder_path: str = '', force_download: bool = False,
                             create_dir=True) -> bool:
    """
    Check whether folder path already exists.

    :param index_name: Name of index
    :param folder_path: Folder path to check
    :param force_download: Flag indicating whether to force download into existing folder
    :param create_dir: Flag indicating whether to create new directory if path does not exist
    :return: Boolean indicating whether to load from file (directory exists)
    """
    if os.path.exists(os.path.join(ROOT_DIR, folder_path)):
        if force_download:
            load_from_file = False
            print('Downloading data from %s into existing folder: %s \n' % (index_name, folder_path))
        else:
            load_from_file = True
            print('Loading data from %s from folder: %s \n' % (index_name, folder_path))
    else:
        if create_dir:
            print('Creating folder for %s: %s' % (index_name, folder_path))
            os.mkdir(os.path.join(ROOT_DIR, folder_path))
        load_from_file = False

    return load_from_file


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


class Timer():
    """
    Timer class for timing operations.
    """

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


class CSVWriter:
    """
    Writer class for handling the logging of training runs into a .csv file.
    """

    def __init__(self, output_path: str, field_names: list):
        self.output_path = output_path
        self.field_names = field_names

        if not os.path.exists(os.path.join(ROOT_DIR, output_path)):
            with open(self.output_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.field_names, delimiter=';', lineterminator='\n')
                writer.writeheader()

    def add_line(self, record: dict):
        """
        Add data line (dict format) to .csv file.

        :param record: Dictionary of records to add to file
        :return:
        """
        with open(self.output_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.field_names, delimiter=';', lineterminator='\n')
            writer.writerow(record)
