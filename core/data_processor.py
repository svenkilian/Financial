import math
import numpy as np
import pandas as pd
from typing import List, Tuple
from colorama import Fore, Back, Style

from utils import pretty_print


class DataLoader:
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, data: pd.DataFrame, cols: list = list, seq_len=None,
                 full_date_range=None, stock_id: tuple = None, split_index: int = 0, model_type: str = '',
                 verbose=False):
        """
        Constructor for DataLoader class

        :param stock_id: Stock identifier tuple
        :param data: DataFrame containing study period data
        :param cols: Columns to use for the model
        :param full_date_range: Full date index of study period
        :param model_type: Type of classifier: 'deep_learning' or 'tree_based'
        """

        # Load data either from csv or pre-loaded DataFrame
        self.model_type = model_type
        self.stock_id = stock_id
        self.seq_len = seq_len
        self.data = data.copy()
        self.cols = cols.copy()
        self.lag_cols = None

        # Handle empty data frame
        if len(self.data) == 0:
            print(f'{Fore.RED}{Style.BRIGHT}Encountered empty DataFrame for index {stock_id}.{Style.RESET_ALL}')
            # TODO: How can this case happen?
            raise AssertionError('Empty DataFrame.')

        # JOB: Determine original split index and split date
        # print(f'Index: {stock_id}')
        self.split_date = full_date_range[split_index]
        # print(f'Original split index: {split_index}, date: {self.split_date.date()}')

        # JOB: Check whether original split date is in index
        if self.split_date.date() in self.data.index:
            # print(f'Split date {self.split_date.date()} in index')
            self.i_split = self.data.index.get_loc(self.split_date.date())
            self.split_date = self.data.index[self.i_split]
        else:
            print(f'Original date ({self.split_date.date()}) not in index.')
            print('Searching for next available split date.')
            try:
                self.i_split = self.data.index.get_loc(self.split_date.date(), method='ffill')
                self.split_date = self.data.index[self.i_split]
            except KeyError as ke:
                print(f'Stock data for {stock_id} does not yield any training data.')
                self.i_split = -1

        # JOB: Apply feature generation if necessary
        if model_type == 'tree_based':
            self.generate_tree_features()

        self.data_train = self.data.loc[:self.split_date, self.cols].values  # Get training array
        self.len_train = len(self.data_train)  # Length of training data

        if self.len_train >= self.seq_len - 1:
            self.data_test = self.data.get(self.cols).iloc[self.i_split - seq_len + 2:].values  # Get test array
            target_start_index = self.i_split + 1
        else:
            self.data_test = self.data.values
            target_start_index = self.i_split + abs(self.i_split - seq_len + 1)

        self.data_test_index = pd.MultiIndex.from_product(
            [self.data.get(self.cols).iloc[target_start_index:].index, [stock_id]])

        self.len_test = len(self.data_test)  # Length of test data
        self.len_train_windows = None

        if verbose:
            print('Length of index: %d' % len(self.data_test_index))
            print('Split index: %s' % self.i_split)
            print('Number of data points: %d' % len(self.data.get(self.cols)))
            print('Number of training data: %d' % self.len_train)
            print('Number of test data: %d' % self.len_test)
            print()

    def get_train_data(self):
        """
        Create x, y training data windows

        Warning:: Batch method, not generative. Make sure you have enough memory to
        load data, otherwise use generate_training_window() method.

        :return:
        """

        data_x = []
        data_y = []

        if self.model_type == 'deep_learning':
            for i in range(self.len_train - self.seq_len + 1):
                x, y = self._next_window(i)
                data_x.append(x)
                data_y.append(y)

            # print('Training data length: %s' % len(data_x))

        elif self.model_type == 'tree_based':
            data_x = self.data_train[self.seq_len - 1:, -len(self.lag_cols):]
            data_y = self.data_train[self.seq_len - 1:, 0].astype(np.int8)

        return np.array(data_x), np.array(data_y)

    def generate_tree_features(self) -> None:
        """
        Generate lagged multi-period returns as features for tree-based methods

        :return:
        """

        # Define lags for multi-period returns
        lags = np.concatenate(
            (np.linspace(1, 20, num=20, dtype=np.int16), np.linspace(40, 240, num=11, dtype=np.int16))).tolist()

        self.lag_cols = [f'm_{lag}_return' for lag in lags]

        for lag in lags:
            # JOB: Recalculate Daily Return
            self.data.loc[:, f'm_{lag}_return'] = self.data.loc[:, 'return_index'].pct_change(periods=lag)

        self.data.drop(columns='return_index', inplace=True)
        self.cols.pop()  # Remove 'return_index' column
        self.cols.extend(self.lag_cols)

    def get_test_data(self, seq_len: int):
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

        if self.model_type == 'deep_learning':
            data_windows = []
            for i in range(self.len_test - seq_len + 1):
                # print(f'Iteration: {i + 1}')
                data_windows.append(self.data_test[i:i + seq_len])
                # print(f'Window length: {len(self.data_test[i:i + seq_len])}')
                # print(f'Last index: {i + seq_len}')

            data_windows = np.array(data_windows).astype(float)

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

        elif self.model_type == 'tree_based':
            x = self.data_test[self.seq_len - 1:, -len(self.lag_cols):]
            y = self.data_test[self.seq_len - 1:, 0].astype(np.int8)

        return x, y

    def _next_window(self, i: int) -> Tuple[np.array, np.array]:
        """
        Generates the next data window from the given index location i

        :param i: Index location
        :return: x, y
        """

        window = self.data_train[i:i + self.seq_len]
        x = window[:-1, 1:]
        y = window[-1, 0].astype(np.int8)

        return x, y

    def generate_train_batch(self, seq_len: int, batch_size: int):
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
                x, y = self._next_window(i)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    # def normalize_windows(self, window_data: np.array, single_window=False):
    #     """
    #     Normalize window with a base value of zero
    #
    #     :param window_data:
    #     :param single_window:
    #     :return:
    #     """
    #
    #     normalized_data = []
    #     window_data = [window_data] if single_window else window_data
    #
    #     for window in window_data:
    #         normalized_window = []
    #         for col_i in range(window.shape[1]):
    #             normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
    #             normalized_window.append(normalised_col)
    #
    #         # Reshape and transpose array back into original multidimensional format
    #         normalized_window = np.array(normalized_window).T
    #         normalized_data.append(normalized_window)
    #
    #     return np.array(normalized_data)
