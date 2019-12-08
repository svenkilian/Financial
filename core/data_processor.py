import math
import numpy as np
import pandas as pd


class DataLoader:
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, filename, split, cols, from_csv=False):
        """
        Constructor for DataLoader class
        :param filename: Filename of csv file or existing DataFrame name
        :param split: Split value between 0 and 1 determining train/test split
        :param cols: Columns to use for the model
        :param from_csv: Load data from .csv file
        """

        # Load data either from csv or pre-loaded DataFrame
        if from_csv:
            dataframe = pd.read_csv(filename)
        else:
            dataframe = filename

        i_split = int(len(dataframe) * split)  # Determine split index
        self.data_train = dataframe.get(cols).values[:i_split]  # Get training array
        self.data_test = dataframe.get(cols).values[i_split:]  # Get test array
        self.len_train = len(self.data_train)  # Length of training data
        self.len_test = len(self.data_test)  # Length of test data
        self.len_train_windows = None

    def get_train_data(self, seq_len, normalize=False):
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

        print('Training data length: %s' % len(data_x))

        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len, normalize=False):
        """
        Create x, y test data windows

        Warning:: Batch method, not generative. Make sure you have enough memory to
        load data, otherwise reduce size of the training split.

        :param seq_len: Sequence length
        :param normalize: Normalize data
        :return:
        """

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]

        print('Test data length: %s' % len(x))

        return x, y

    def generate_train_batch(self, seq_len, batch_size, normalize):
        """
        Yield a generator of training data from filename on given list of cols split for train/test

        :param seq_len: Sequence length
        :param batch_size: Batch size
        :param normalize: Normalize data
        :return:
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

    def _next_window(self, i, seq_len, normalize):
        """
        Generates the next data window from the given index location i

        :param i: Index location
        :param seq_len: Sequence length
        :param normalize: Normalize data
        :return: x, y
        """

        window = self.data_train[i:i + seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1]
        y = window[-1, [0]]

        return x, y

    def normalize_windows(self, window_data, single_window=False):
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
