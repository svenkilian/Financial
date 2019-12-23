import math
import numpy as np
import pandas as pd
from typing import List, Tuple


class DataLoader:
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, data: pd.DataFrame, split: float = 0.75, cols: list = list, from_csv=False, seq_len=None,
                 full_date_range=None):
        """
        Constructor for DataLoader class
        :param data: DataFrame containing study period data
        :param split: Split value between 0 and 1 determining train/test split
        :param cols: Columns to use for the model
        :param from_csv: Load data from .csv file
        """

        # Load data either from csv or pre-loaded DataFrame
        if from_csv:
            dataframe = pd.read_csv(data)
        else:
            dataframe = data

        i_split = int(len(full_date_range) * split)  # Determine split index
        split_date = full_date_range[i_split]
        # print('Split index: %d' % i_split)
        # print('Split date: %s' % split_date)
        self.data_train = dataframe.get(cols).loc[:split_date].values  # Get training array
        # TODO: Handle error of non-existent data access
        self.data_test = dataframe.get(cols).loc[full_date_range[i_split - seq_len]:].values  # Get test array
        # TODO: ----------------------------------------
        self.data_test_index = dataframe.loc[full_date_range[i_split - seq_len]:].index
        self.len_train = len(self.data_train)  # Length of training data
        self.len_test = len(self.data_test)  # Length of test data
        self.len_train_windows = None

        print(len(self.data_test_index))
        # print('Split index: %s' % i_split)
        # print('Number of data points: %d' % len(dataframe))
        # print('Number of training data: %d' % self.len_train)
        # print('Number of test data: %d' % self.len_test)

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

        data_windows = []
        for i in range(self.len_test - seq_len + 1):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:, :-1, 1:]
        y = data_windows[:, -1, [0]]

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
