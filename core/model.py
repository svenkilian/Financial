GPU_ENABLED = True

import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
import pprint
from tensorflow.keras import optimizers, regularizers
import tensorflow as tf

from utils import Timer
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.ensemble import RandomForestClassifier

# tf.logging.set_verbosity(tf.logging.ERROR)


class LSTMModel:
    """
    LSTM model class
    """

    def __init__(self, index_name=None):
        self.model = Sequential()  # Initialize Keras sequential model
        if index_name:
            self.index_name = index_name
        else:
            self.index_name = ''

    def __repr__(self):
        return str(self.model.summary())

    def load_model(self, filepath):
        """
        Load model from file
        :param filepath: File name
        :return:
        """
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, verbose=2):
        """
        Build model from configuration file

        :param verbose: Verbosity of console output
        :param configs:
        :return:
        """
        timer = Timer()
        timer.start()

        layer_type_dict = {key.lower(): value for key, value in layers.__dict__.items()}
        if '_dw_wrapped_module' in layer_type_dict.keys():
            layer_type_dict = {key.lower(): value for key, value in
                               layer_type_dict.get('_dw_wrapped_module').__dict__.items()}

        for layer in configs['model']['layers']:
            # Get layer type string
            layer_type = layer['type'].lower()
            # Add layer with configuration
            self.model.add(layer_type_dict[layer_type].from_config(layer['params']))

            # if layer['type'] == 'lstm':
            #     # self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), dropout=dropout,
            #     #                     return_sequences=return_seq))
            #     self.model.add(LSTM(**layer['params']))

        pprinter = pprint.PrettyPrinter(indent=2)
        if verbose == 1:
            for i, layer in enumerate(self.model.layers):
                print(f'Layer {i + 1} ({layer.name.upper()}):')
                pprinter.pprint(layer.get_config())
                print()

        self.model.compile(loss=configs['model']['loss'],
                           optimizer=get_optimizer(configs['model']['optimizer'], parameters=
                           configs['model']['optimizer_params']),
                           metrics=configs['model']['metrics'])

        if verbose == 1:
            print(f'\nOptimizer configuration: ')
            pprinter.pprint(self.model.optimizer.get_config())
            print()
        if verbose != 0:
            print(self.model.summary())
            print()

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: int, save_dir: str, configs: dict,
              verbose=1):
        """
        Train model (in-memory)

        :param x: Input data
        :param y: Target data
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param save_dir: Path to model directory
        :param configs: Configuration dict
        :param verbose: Verbosity

        :return: Training history
        """

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        early_stopping_patience = configs['training']['early_stopping_patience']

        save_fname = os.path.join(save_dir, '%s-%s-e%s-b%s.h5' % (self.index_name,
                                                                  dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
                                                                  str(epochs), str(batch_size)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True,
                          verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True, verbose=1)]

        if GPU_ENABLED:
            previous_runs = os.listdir('logs')
            # print(f'Directory contents: {previous_runs}')
            if len(previous_runs) == 0:
                run_number = 1
            else:
                run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
            log_dir_name = 'run_%02d' % run_number
            print(f'Log directory: {os.path.join("logs", log_dir_name)}')
            callbacks.append(
                TensorBoard(log_dir=os.path.join('logs', log_dir_name), histogram_freq=0, write_graph=False,
                            write_grads=False,
                            write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None,
                            embeddings_data=None, update_freq='epoch'))

        try:
            history = self.model.fit(
                x,
                y,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_split=0.20,
                shuffle=True,
                verbose=verbose
            )
        except TypeError as te:
            print(te)
            history = None

        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

        return history

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        """
        Train model using generator

        :param data_gen: Generator object
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param steps_per_epoch: Steps per epoch
        :param save_dir: Path to saving directory
        :return:
        """

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data):
        """
        Predict each time step given the last sequence of true data, in effect only predicting 1 step ahead each time

        :param data:
        :return:
        """
        print('[Model] Predicting Point-by-Point ...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """
        Predict sequence of 50 steps before shifting prediction run forward by 50 steps

        :param data:
        :param window_size:
        :param prediction_len:
        :return:
        """

        print('[Model] Predicting Sequences Multiple ...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        """
        Shift the window by 1 new prediction each time, re-run predictions on new window

        :param data:
        :param window_size:
        :return:
        """

        print('[Model] Predicting Sequences Full ...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted


def get_optimizer(optimizer_name: str, parameters: dict = None):
    """
    Retrieve keras optimizer for a given optimizer name

    :param optimizer_name: Optimizer name
    :param parameters: Parameters specifying optimizer options
    :return: Parametrized keras optimizer
    """

    try:
        optimizer = optimizers.get(optimizer_name).from_config(parameters)

    except ValueError:
        print('Specified optimizer name is unknown.')
        print('Resorting to RMSprop as default.')
        optimizer = optimizers.RMSprop()

    return optimizer


class RandomForestModel:
    """
    Random Forest model class
    """

    def __init__(self, index_name):

        self.model = None
        if index_name:
            self.index_name = index_name
        else:
            index_name = ''

    def __repr__(self):
        return str(self.model.get_params())

    def build_model(self, verbose=2):
        parameters = {'n_estimators': 1000,
                      'max_depth': 20,
                      'n_jobs': -1,
                      'verbose': 1}
        self.model = RandomForestClassifier(**parameters)
        if verbose == 2:
            print(type(self.model).__name__)
        elif verbose == 1:
            print(self.model)


