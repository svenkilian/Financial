import math
import numpy as np
import pandas as pd
from typing import List, Tuple
from colorama import Fore, Back, Style

from utils import pretty_print


class DataLoader:
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, data: pd.DataFrame, split: float = 0.75, cols: list = list, seq_len=None,
                 full_date_range=None, stock_id: tuple = None, split_index: int = 0):
        """
        Constructor for DataLoader class

        :param stock_id: Stock identifier tuple
        :param data: DataFrame containing study period data
        :param split: Split value between 0 and 1 determining train/test split
        :param cols: Columns to use for the model
        :param full_date_range: Full date index of study period
        """

        # Load data either from csv or pre-loaded DataFrame
        self.stock_id = stock_id
        self.data = data

        # Handle empty data frame
        if len(self.data) == 0:
            print(f'{Fore.RED}{Style.BRIGHT}Encountered empty DataFrame for index {stock_id}.{Style.RESET_ALL}')
            # TODO: How can this case happen?
            raise AssertionError('Empty DataFrame.')

        # JOB: Determine original split index and split date
        # print(f'Index: {stock_id}')
        split_date = full_date_range[split_index]
        # print(f'Original split index: {split_index}, date: {split_date.date()}')

        # JOB: Check whether original split date is in index
        if split_date.date() in self.data.index:
            # print(f'Split date {split_date.date()} in index')
            i_split = self.data.index.get_loc(split_date.date())
            split_date = self.data.index[i_split]
        else:
            print(f'Original date ({split_date.date()}) not in index.')
            print('Searching for next available split date.')
            try:
                i_split = self.data.index.get_loc(split_date.date(), method='ffill')
                split_date = self.data.index[i_split]
            except KeyError as ke:
                print(f'Stock data for {stock_id} does not yield any training data.')
                i_split = -1

        # print(f'Using {split_date.date()} as split date.')

        self.data_train = self.data.loc[:split_date, cols].values  # Get training array

        if i_split - seq_len + 2 >= 0:
            self.data_test = self.data.get(cols).iloc[i_split - seq_len + 2:].values  # Get test array
            target_start_index = i_split + 1
        else:
            self.data_test = self.data.get(cols).values
            target_start_index = i_split + abs(i_split - seq_len + 1)

        self.data_test_index = pd.MultiIndex.from_product(
            [self.data.get(cols).iloc[target_start_index:].index, [stock_id]])

        self.len_train = len(self.data_train)  # Length of training data
        self.len_test = len(self.data_test)  # Length of test data
        self.len_train_windows = None

        # print('Length of index: %d' % len(self.data_test_index))
        # print('Split index: %s' % i_split)
        # print('Number of data points: %d' % len(self.data.get(cols)))
        # print('Number of training data: %d' % self.len_train)
        # print('Number of test data: %d' % self.len_test)
        # print()

    def get_train_data(self, seq_len: int, normalize=False):
        """
        Create x, y training data windows

        Warning:: Batch method, not generative. Make sure you have enough memory to
        load data, otherwise use generate_training_window() method.

        :param seq_len: Sequence length
        :param normalize: normalize data
        :return:
        """

        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)

        # print('Training data length: %s' % len(data_x))

        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len: int, normalize=False):
        """
        Create x, y test data windows

        Warning:: Batch method, not generative. Make sure you have enough memory to
        load data, otherwise reduce size of the training split.

        :param seq_len: Sequence length
        :param normalize: Normalize data
        :return:
        """

        x = None
        y = None

        data_windows = []
        for i in range(self.len_test - seq_len + 1):
            # print(f'Iteration: {i + 1}')
            data_windows.append(self.data_test[i:i + seq_len])
            # print(f'Window length: {len(self.data_test[i:i + seq_len])}')
            # print(f'Last index: {i + seq_len}')

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        # print(data_windows)
        if len(data_windows) > 0:
            x = data_windows[:, :-1, 1:]
            y = data_windows[:, -1, [0]]
        else:
            # print(self.stock_id)
            # print(self.len_test)
            # print(self.len_train)
            x = np.array([])
            y = np.array([])
            print(
                f'{Fore.RED}{Style.BRIGHT}Non-positive test data length for {self.stock_id}.{Style.RESET_ALL}')

        # print('Test data length: %s' % len(x))
        # print()

        return x, y

    def generate_train_batch(self, seq_len: int, batch_size: int, normalize: bool):
        """
        Yield a generator of training data from filename on given list of cols split for train/test

        :param seq_len: Sequence length
        :param batch_size: Batch size
        :param normalize: Normalize data
        :return:

        Usage::

        # Out-of memory generative training
        steps_per_epoch = math.ceil(
                        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']
                        )

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

        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # Stop condition for a smaller final batch if data does not divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalize)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i: int, seq_len: int, normalize: bool) -> Tuple[np.array, np.array]:
        """
        Generates the next data window from the given index location i

        :param i: Index location
        :param seq_len: Sequence length
        :param normalize: Normalize data
        :return: x, y
        """

        window = self.data_train[i:i + seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1, 1:]
        y = window[-1, [0]]

        return x, y

    def normalize_windows(self, window_data: np.array, single_window=False):
        """
        Normalize window with a base value of zero

        :param window_data:
        :param single_window:
        :return:
        """

        normalized_data = []
        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalised_col)

            # Reshape and transpose array back into original multidimensional format
            normalized_window = np.array(normalized_window).T
            normalized_data.append(normalized_window)

        return np.array(normalized_data)
