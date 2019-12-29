"""
This utilities module implements helper functions for displaying data frames and plotting data
"""
import datetime as dt
import json

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import ticker
from pandas.plotting import register_matplotlib_converters
import glob
import os

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
            label_mapping=indices, export_as_png=True)
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
    Prints out DataFrame in tabular format

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
    Plot training and validation metrics over epochs

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
        fig.savefig(os.path.join(folder_path, 'metrics.png'), dpi=600)

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
        fig.savefig(os.path.join(folder_path, 'loss.png'), dpi=600)


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def get_most_recent_file(directory: str) -> str:
    all_files = glob.glob(directory + '/*')
    most_recent_file = max(all_files, key=os.path.getctime)

    # print(all_files)
    # print(most_recent_file)

    return most_recent_file


def lookup_multiple(dict_of_dicts: dict = None, data_folder: str = '', index_id: str = ''):
    # Load index name dict and get index name
    for key, value in dict_of_dicts.items():
        print(f'Trying lookup in {key} ...')
        lookup_dict = json.load(open(os.path.join(data_folder, 'data', value.get('file_path')), 'r'))
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


class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
