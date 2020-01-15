import inspect
from typing import List

from sklearn import ensemble

from config import ROOT_DIR

GPU_ENABLED = True

import os
import numpy as np
import datetime as dt
from numpy import newaxis
import pprint
from tensorflow.keras import optimizers
from colorama import Fore, Back, Style

from core.utils import Timer
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree


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
        self.configs = None

    def __repr__(self):
        return str(self.model.summary())

    def get_params(self):
        return self.configs

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

        self.configs = configs

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

        save_fname = os.path.join(ROOT_DIR, save_dir, '%s-%s-e%s-b%s.h5' % (self.index_name,
                                                                            dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
                                                                            str(epochs), str(batch_size)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True,
                          verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True, verbose=1)]

        if GPU_ENABLED:
            previous_runs = os.listdir(os.path.join(ROOT_DIR, 'logs'))
            # print(f'Directory contents: {previous_runs}')
            if len(previous_runs) == 0:
                run_number = 1
            else:
                run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
            log_dir_name = 'run_%02d' % run_number
            print(f'Log directory: {os.path.join("logs", log_dir_name)}')
            callbacks.append(
                TensorBoard(log_dir=os.path.join(ROOT_DIR, 'logs', log_dir_name), histogram_freq=0, write_graph=False,
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

        save_fname = os.path.join(ROOT_DIR, save_dir,
                                  '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
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

    def predict_point_by_point(self, data) -> np.array:
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


class TreeEnsemble:
    """
    Random Forest model class
    """

    def __init__(self, index_name=None, model_type='RandomForestClassifier'):

        self.model = None
        self.model_type = model_type
        if index_name:
            self.index_name = index_name
        else:
            index_name = ''
        self.parameters = None

    def __repr__(self):
        return str(self.model)

    def get_params(self):
        return self.model.get_params()

    def build_model(self, configs: dict, verbose=2):

        # Create dictionary of sklearn ensembles
        ensemble_clfs = ensemble.__dict__['__all__']
        ensemble_type_dict = {key.lower(): ensemble.__dict__.get(key) for key in ensemble_clfs if
                              inspect.isclass(ensemble.__dict__.get(key))}

        # Retrieve model parameters from config
        self.parameters = configs[self.model_type]

        if 'base_estimator' in self.parameters.keys():
            base_estimator_name = self.parameters.get('base_estimator')
            base_clfs = sklearn.tree.__dict__['__all__']
            base_type_dict = {key.lower(): sklearn.tree.__dict__.get(key) for key in base_clfs if
                              inspect.isclass(sklearn.tree.__dict__.get(key))}
            base_estimator = base_type_dict[base_estimator_name.lower()]()
            self.parameters['base_estimator'] = base_estimator

        # Extract nested parameters
        nested_params = {key: val for key, val in self.parameters.items() if '__' in key}

        # Filter for non-nested parameters
        for nested_parameter in nested_params.keys():
            self.parameters.pop(nested_parameter)

        # Instantiate model
        self.model = ensemble_type_dict[self.model_type.lower()](**self.parameters)
        self.model.set_params(**nested_params)

        if 'verbose' in self.model.get_params().keys():
            self.parameters['verbose'] = verbose
            self.model.set_params(verbose=verbose)

        if verbose == 2:
            print(type(self.model).__name__)
        elif verbose == 1:
            print(f'{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{self.model}{Style.RESET_ALL}')

        return self


class WeightedEnsemble:
    """
    Class representing a weighted ensemble of individual classifiers
    """

    def __init__(self, index_name: str = '', classifier_type_list: List[str] = None, configs: dict = None):
        self.classifier_types = classifier_type_list
        self.classifiers = [TreeEnsemble(index_name=index_name, model_type=m_type).build_model(configs) for m_type in
                            self.classifier_types]
        self.oob_scores = list()

    def __repr__(self):
        out = []
        for model in self.classifiers:
            out.append(str(model))

        return str(out)

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fit individual models

        :return:
        """

        for model in self.classifiers:
            print(f'\n\nFitting {model.model_type} model ...')
            timer = Timer().start()
            model.model.fit(x_train, y_train)
            self.oob_scores.append(model.model.oob_score_)  # TODO: Move somewhere else

            print('Feature importances:')
            print(model.model.feature_importances_)
            timer.stop()

    def get_params(self):
        """

        :return:
        """
        params = []
        for model in self.classifiers:
            params.append(model.get_params())
        return params

    def predict_all(self, x_test: np.array) -> np.array:
        """
        Predict on test set, separately for each individual classifier

        :param x_test: Test set

        :return: Array with individual predictions in rows
        """
        predictions_set = []

        for model in self.classifiers:
            predictions = model.model.predict_proba(x_test)[:, 1]
            predictions_set.append(predictions)

        return np.array(predictions_set)

    def predict(self, x_test: np.array, weighted=False, oob_scores: list = None, alpha=2) -> np.array:
        """
        Aggregate individual predictions

        :param alpha: Weighting parameter for weighted average
        :param oob_scores: Out-of-bag performance score of base classifiers
        :param weighted: Use performance-weighted average
        :param x_test: Test set
        :return:
        """
        print('Making aggregated predictions ...')
        timer = Timer().start()
        if weighted:
            print(f'Weighting with OOB scores [{", ".join([str(np.round(score, 3)) for score in oob_scores])}')
            weights = [score ** alpha / sum([sc ** alpha for sc in oob_scores]) for score in oob_scores]
            print(f'Weights: [{", ".join([str(weight) for weight in weights])}')
            predictions = np.average(self.predict_all(x_test), weights=weights, axis=0)
        else:
            predictions = np.mean(self.predict_all(x_test), axis=0)
        timer.stop()

        return predictions
