"""
This module implements the machine learning models used in the study
"""

import inspect

from scipy.stats import rankdata
from sklearn import ensemble

from config import ROOT_DIR
from core import execute

GPU_ENABLED = True

import os
import numpy as np
import datetime as dt
from numpy import newaxis, ndarray
import pandas as pd
import pprint
from tensorflow.keras import optimizers
from colorama import Fore, Style
from typing import List, Tuple

from core.utils import Timer, get_model_parent_type, pretty_print_table, to_combinations
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import sklearn.tree


# tf.logging.set_verbosity(tf.logging.ERROR)


#

class LSTMModel:
    """
    LSTM model class
    """

    def __init__(self):
        self.model = Sequential()  # Initialize Keras sequential model
        self.configs = None
        self.model_type = 'LSTM'
        self.parent_model_type = 'deep_learning'

    def __repr__(self):
        return f'{Style.BRIGHT}{Fore.RED}{self.model.__class__.__name__}' \
               f'({", ".join([layer.__class__.__name__ for layer in self.model.layers])}){Style.RESET_ALL}'

    def get_params(self):
        return {'data': self.configs['data'], 'training': self.configs['training'], 'model': self.configs['model']}

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
            if layer.get('params').get('input_shape'):
                layer['params']['input_shape'] = (
                    layer['params']['input_shape'][0], len(self.configs.get('data').get('columns')))
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

        return self

    def fit(self, x_train: np.array, y_train: np.array, verbose=1, **fit_args):
        validation_data = None
        if fit_args.get('x_val') is not None:
            # In case classifier is part of mixed ensembe, extract valudation data
            validation_data = (fit_args['x_val'][fit_args['model_index']], fit_args['y_test'][fit_args['model_index']])
            validation_split = None
        else:
            validation_split = 0.2

        history = self.train(
            x_train,
            y_train,
            epochs=self.configs['training']['epochs'],
            batch_size=self.configs['training']['batch_size'],
            save_dir=self.configs['model']['save_dir'], configs=self.configs, validation_data=validation_data,
            validation_split=validation_split,
            verbose=verbose)

        if fit_args.get('x_val'):
            val_score = execute.test_model(predictions=self.predict(fit_args.get('x_val')[fit_args['model_index']]),
                                           get_val_score_only=True, model=self, **fit_args)
            return val_score

        best_val_accuracy = float(history.history['val_accuracy'][np.argmin(history.history['val_loss'])])

        return best_val_accuracy

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: int, save_dir: str, configs: dict,
              validation_data: Tuple[np.array, np.array] = None, validation_split=0.2,
              verbose=1):
        """
        Train model (in-memory)

        :param validation_split:
        :param validation_data:
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
        print('[LSTM Model] Training Started')
        print(f'[LSTM Model] Epochs: {epochs}, batch size: {batch_size} ')
        early_stopping_patience = configs['training']['early_stopping_patience']

        save_fname = os.path.join(ROOT_DIR, save_dir, '%s-%s-e%s-b%s.h5' % ('training_',
                                                                            dt.datetime.now().strftime('%d%m%Y-%H%M%S'),
                                                                            str(epochs), str(batch_size)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True,
                          verbose=1, min_delta=0.00005),
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
                validation_split=validation_split,
                validation_data=validation_data,
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

    def predict(self, x_test: np.array) -> np.array:
        """
        Predict each time step given the last sequence of true data, in effect only predicting 1 step ahead each time

        :param x_test: Test data
        :return: Predictions
        """
        print('[LSTM Model] Predicting on test data ...')
        predicted = self.model.predict(x_test)
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

    def __init__(self, model_type='RandomForestClassifier'):

        self.model = None
        self.model_type = model_type
        self.parent_model_type = 'tree_based'
        self.parameters = None

    def __repr__(self):
        # return f'{Style.BRIGHT}{Fore.YELLOW}{self.model}{Style.RESET_ALL}'
        return f'{Style.BRIGHT}{Fore.YELLOW}{self.model_type}{Style.RESET_ALL}'

    def fit(self, x_train: np.array, y_train: np.array, **fit_args):
        """
        Fit TreeEnsemble to training data
        :param x_train: Training data features
        :param y_train: Training data targets
        :param fit_args: Optional fitting arguments for obtaining validation score
        :return: Validation score / OOB score, respectively
        """
        self.model.fit(x_train, y_train)
        if fit_args.get('x_val'):
            # In case TreeEnsemble is part of MixedEnsemble, get validation scores
            val_score = execute.test_model(
                predictions=self.model.predict_proba(fit_args.get('x_val')[fit_args['model_index']])[:, 1],
                get_val_score_only=True, model=self, **fit_args)
            return val_score
        ret_val = None
        if hasattr(self.model, 'oob_score_'):
            ret_val = self.model.oob_score_
        return ret_val

    def get_params(self):
        return str(self.model.get_params())

    def build_model(self, configs: dict, verbose=2):

        # Create dictionary of sklearn ensembles
        ensemble_clfs = ensemble.__dict__['__all__']
        ensemble_type_dict = {key.lower(): ensemble.__dict__.get(key) for key in ensemble_clfs if
                              inspect.isclass(ensemble.__dict__.get(key))}

        # Retrieve model parameters from config
        self.parameters = configs['tree_based_classifiers'][self.model_type]

        if 'base_estimator' in self.parameters.keys():
            base_estimator_name = self.parameters.get('base_estimator')
            base_clfs = sklearn.tree.__dict__['__all__']
            base_type_dict = {key.lower(): sklearn.tree.__dict__.get(key) for key in base_clfs if
                              inspect.isclass(sklearn.tree.__dict__.get(key))}
            if isinstance(base_estimator_name, str):
                base_estimator = base_type_dict[base_estimator_name.lower()]()
            else:
                base_estimator = base_estimator_name
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

    def __init__(self, index_name: str = '', classifier_type_list: List[str] = None, configs: dict = None,
                 verbose: int = 0):
        """

        :rtype: object
        """
        self.classifier_types = classifier_type_list
        self.classifiers = [TreeEnsemble(model_type=m_type).build_model(configs, verbose=verbose)
                            for m_type in
                            self.classifier_types]
        self.model_type = f'{self.__class__.__name__}({", ".join([str(clf.model_type) for clf in self.classifiers])})'
        self.oob_scores = list()

    def __repr__(self):
        out = f'{Style.BRIGHT}{Fore.BLUE}{self.__class__.__name__}' \
              f'({", ".join([str(clf.model_type) for clf in self.classifiers])}){Style.RESET_ALL}'

        return out

    def fit(self, x_train: np.array, y_train: np.array, feature_names: List[str] = None, show_importances=True):
        """
        Fit individual models

        :return:
        """

        for model in self.classifiers:
            print(f'\n\nFitting {Style.BRIGHT}{Fore.BLUE}{model.model_type}{Style.RESET_ALL} model ...')
            timer = Timer().start()
            model.model.fit(x_train, y_train)
            if hasattr(model.model, 'oob_score_'):
                self.oob_scores.append(model.model.oob_score_)

            if show_importances:
                if feature_names:
                    feature_importances = pd.DataFrame(
                        {'Feature Importance': model.model.feature_importances_.tolist()},
                        index=feature_names)
                else:
                    feature_importances = model.model.feature_importances_

                print('Feature importances:')
                pretty_print_table(feature_importances)
            timer.stop()

    def get_params(self):
        """

        :return:
        """
        params = []
        for model in self.classifiers:
            params.append(str(model.get_params()))
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
            print(f'Weighting with OOB scores [{", ".join([str(np.round(score, 3)) for score in oob_scores])}]')
            weights = [max(score - .5, 0) ** alpha / sum([max(sc - .5, 0) ** alpha for sc in oob_scores]) for score in
                       oob_scores]
            print(f'Weights: [{", ".join([str(weight) for weight in weights])}]')
            predictions = np.average(self.predict_all(x_test), weights=weights, axis=0)
        else:
            predictions = np.mean(self.predict_all(x_test), axis=0)
        timer.stop()

        return predictions

    @staticmethod
    def is_weighted_ensemble(classifier_list, configs):
        """
        Determine whether list of classifiers qualified as weighted tree-based ensemble

        :param classifier_list: List of classifier candidates
        :param configs: Configurations
        :return: Is mixed type
        """

        return all([clf in configs['model_hierarchy']['tree_based'] for clf in classifier_list])


class MixedEnsemble:
    """
    Implements a mixed ensemble with a LSTM and a tree-based classifier
    """

    def __init__(self, index_name: str = '', classifier_type_list: List[str] = None, configs: dict = None,
                 weighting_criterion='Accuracy', verbose: int = 0, mute=False):
        self.index_name = index_name
        self.classifier_types = classifier_type_list
        self.configs = configs
        self.classifiers = [
            TreeEnsemble(model_type=m_type).build_model(self.configs, verbose=verbose) if
            get_model_parent_type(model_type=m_type) == 'tree_based'
            else LSTMModel().build_model(configs=configs,
                                         verbose=verbose) for m_type in
            self.classifier_types] if classifier_type_list is not None else []
        self.__val_scores = []
        self.weighting_criterion = weighting_criterion
        self.verbose = verbose
        if not mute:
            print(f'Successfully created mixed ensemble of {[clf for clf in self.classifiers]}')

    @classmethod
    def from_classifier_list(cls, classifiers: list, classifier_types: list, val_scores: list, weighting_criterion,
                             configs, verbose):
        model = cls(configs=configs, weighting_criterion=weighting_criterion, mute=True)
        model.classifier_types = classifier_types
        model.val_scores = val_scores
        model.classifiers = list(classifiers)
        model.weighting_criterion = weighting_criterion
        model.verbose = verbose
        print(f'Successfully created mixed ensemble of {[clf for clf in model.classifiers]}')

        return model

    def __repr__(self):
        out = f'{Style.BRIGHT}{Fore.BLUE}{self.__class__.__name__}' \
              f'({", ".join([str(clf.model_type) for clf in self.classifiers])}){Style.RESET_ALL}'

        return out

    @property
    def val_scores(self):
        return self.__val_scores

    @val_scores.setter
    def val_scores(self, value):
        self.__val_scores = value

    @property
    def combinations(self):
        sub_ensembles = []
        clf_combinations, clf_type_combinations, clf_val_combinations = to_combinations(
            self.classifiers), to_combinations(self.classifier_types), to_combinations(self.val_scores)

        for i in range(len(clf_combinations)):
            sub_ensembles.append(MixedEnsemble.from_classifier_list(classifiers=list(clf_combinations[i]),
                                                                    classifier_types=list(clf_type_combinations[i]),
                                                                    val_scores=list(clf_val_combinations[i]),
                                                                    weighting_criterion=self.weighting_criterion,
                                                                    configs=self.configs, verbose=self.verbose))
        return sub_ensembles

    @property
    def model_type(self):
        return f'{self.__class__.__name__}({", ".join([str(clf.model_type) for clf in self.classifiers])})'

    @property
    def data_set_indices(self):
        return [0 if model.parent_model_type == 'deep_learning' else 1 for model in self.classifiers]

    def fit(self, x_train: list, y_train: list, feature_names: List[List[str]] = None, **fit_args) -> list:
        """
        Fit model to training data
        :param x_train: Training features
        :param y_train: Training target
        :param feature_names: List of feature names
        :param fit_args: Fitting arguments for obtaining validation scores

        :return: Validation scores of fitted model
        """

        for i, model in enumerate(self.classifiers):
            # JOB: Fit each ensemble component and evaluate performance on validation data
            print(f'\n\nFitting {Style.BRIGHT}{Fore.BLUE}{model.model_type}{Style.RESET_ALL} model ...')
            data_set_index = None
            if model.parent_model_type == 'deep_learning':
                data_set_index = 0
            elif model.parent_model_type == 'tree_based':
                data_set_index = 1
            timer = Timer().start()
            val_score = model.fit(x_train[data_set_index], y_train[data_set_index], model_index=data_set_index,
                                  verbose=self.verbose, model_type=model.model_type, **fit_args)
            if val_score:
                self.val_scores.append(val_score)

            timer.stop()

        return self.val_scores

    def get_params(self):
        pass

    def predict_all(self, x_test: np.array, y_test: np.array, test_data_index: ndarray) -> Tuple:
        """
        Predict on test set, separately for each individual classifier

        :param y_test:
        :param test_data_index:
        :param x_test: Test set

        :return: (Array with individual predictions in rows, merged index)
        """
        predictions_set = []
        predictions_index_set = []
        y_test_sets = []
        index_merged = None
        y_test_indexed_merged = None

        # Build test set indices for individual predictions
        for model_index, _ in enumerate(self.classifiers):
            predictions_index = pd.DataFrame({f'predictions_{model_index}': 0},
                                             index=pd.MultiIndex.from_tuples(
                                                 test_data_index[self.data_set_indices[model_index]],
                                                 names=['datadate', 'stock_id']))
            predictions_index_set.append(predictions_index)

            y_test_indexed = pd.DataFrame({f'y_test_{model_index}': y_test[self.data_set_indices[model_index]].ravel()},
                                          index=pd.MultiIndex.from_tuples(test_data_index[self.data_set_indices[
                                              model_index]],
                                                                          names=['datadate', 'stock_id']))
            y_test_sets.append(y_test_indexed)

        for i in range(len(predictions_index_set) - 1):
            if i == 0:
                index_merged = predictions_index_set[i]
                y_test_indexed_merged = y_test_sets[i]

            index_merged = index_merged.merge(predictions_index_set[i + 1], how='inner',
                                              left_index=True,
                                              right_index=True, suffixes=(False, False))
            y_test_indexed_merged = y_test_indexed_merged.merge(y_test_sets[i + 1], how='inner', left_index=True,
                                                                right_index=True, suffixes=(False, False))

        y_test_indexed_merged = y_test_indexed_merged.iloc[:, 0]

        index_merged = index_merged.index
        print(f'Length of merged index: {len(index_merged)}')  # TODO: Remove

        for model_index, model in enumerate(self.classifiers):
            i = None
            if model.parent_model_type == 'deep_learning':
                i = 0
            elif model.parent_model_type == 'tree_based':
                i = 1
            predictions = None
            if isinstance(model, TreeEnsemble):
                predictions = model.model.predict_proba(x_test[i])[:, 1]
            elif isinstance(model, LSTMModel):
                predictions = model.predict(x_test[i])

            predictions_indexed = pd.DataFrame({f'predictions_{i}': predictions},
                                               index=pd.MultiIndex.from_tuples(
                                                   test_data_index[self.data_set_indices[i]],
                                                   names=['datadate', 'stock_id']))

            predictions_indexed = predictions_indexed.loc[index_merged]
            # print(f'Length of indexed prediction: {len(predictions_indexed)}')
            # print(predictions_indexed.iloc[:, 0].values)
            predictions_set.append(predictions_indexed.iloc[:, 0].values)

        print(', '.join([str(len(l)) for l in predictions_set]))
        if len(set(([len(preds) for preds in predictions_set]))) != 1:
            raise AssertionError('Test set sizes do not match.')

        return np.array(predictions_set), y_test_indexed_merged.values, index_merged

    def predict(self, x_test: ndarray, y_test_merged: ndarray, all_predictions: ndarray = None,
                test_data_index: ndarray = None,
                test_data_index_merged: pd.MultiIndex = None,
                y_test: ndarray = None, weighted=True, performance_based=True, rank_based=True, equal_weights=True,
                alpha=2) -> np.array:
        """
        Aggregate individual predictions to weighted predictions according to weighting schemes

        :param alpha: Weighting parameter for weighted average
        :param performance_based: Use performance-based weighting
        :param equal_weights: Use equal weights weighting
        :param rank_based: Use rank-based weighting
        :param y_test_merged: Test target array
        :param test_data_index_merged: Test data index
        :param all_predictions: All predictions
        :param y_test: Individual-level test targets
        :param test_data_index: Array of test data indices
        :param weighted: Use performance-weighted average
        :param x_test: Test set
        :return: Tuple(predictions, y_test_indexed_merged.values, predictions_indexed_merged.index
        """
        print('Making aggregated predictions ...')

        if all_predictions is not None:
            pass
        else:
            all_predictions, y_test_merged, test_data_index_merged = self.predict_all(x_test, y_test=y_test,
                                                                                      test_data_index=test_data_index)

        timer = Timer().start()

        if weighted:
            predictions = []  # Initialize list of predictions
            if performance_based:
                # JOB: Aggregate predictions based on the component models' validation score
                print(
                    f'\nPerformance-based weighting with validation scores ({self.weighting_criterion}): [{", ".join([str(np.round(score, 4)) for score in self.val_scores])}]')
                try:
                    weights = [max(score, 0) ** alpha / max(sum([max(sc, 0) ** alpha for sc in self.val_scores]), 0) for
                               score
                               in
                               self.val_scores]
                except ZeroDivisionError as zde:
                    print(
                        f'{Style.BRIGHT}{Fore.LIGHTRED_EX}All validation scores are <= 0. '
                        f'Resorting to equal weighting.{Style.RESET_ALL}')
                    weights = [1 / len(self.val_scores) for score in self.val_scores]
                print(f'Weights: [{", ".join([str(round(weight, 3)) for weight in weights])}]')
                preds = np.average(all_predictions, weights=weights, axis=0)
                predictions.append(('performance', preds))
            if rank_based:
                # JOB: Aggregate predictions based on the component models' validation score rank
                print(
                    f'\nRank-based weighting with validation scores ({self.weighting_criterion}): '
                    f'[{", ".join([str(np.round(score, 4)) for score in self.val_scores])}]')

                ranks = (len(self.val_scores) + 1) - rankdata(self.val_scores, method='ordinal')
                weights = [(1 / rank_i) / sum([(1 / rank_j) for rank_j in ranks]) for rank_i in ranks]

                print(f'Weights: [{", ".join([str(round(weight, 3)) for weight in weights])}]')
                preds = np.average(all_predictions, weights=weights, axis=0)
                predictions.append(('rank', preds))
            if equal_weights:
                # JOB: Aggregate predictions using equal weights
                print('\nWeighting with equal weights:')
                weights = [1 / len(self.classifier_types) for _ in self.classifier_types]
                print(f'Weights: [{", ".join([str(round(weight, 3)) for weight in weights])}]')
                preds = np.mean(all_predictions, axis=0)
                predictions.append(('equal', preds))

        else:
            # JOB: In case no weighting scheme is specified, resort to equal weighting
            predictions = np.mean(all_predictions, axis=0)
        timer.stop()

        return predictions, y_test_merged, test_data_index_merged

    @staticmethod
    def is_mixed_ensemble(classifier_list, configs):
        """
        Determine whether list of classifiers qualified as mixed ensemble

        :param classifier_list: List of classifier candidates
        :param configs: Configurations
        :return: Is mmixed type
        """

        has_tree_based = len(set(classifier_list).intersection(configs['model_hierarchy']['tree_based'])) > 0
        has_lstm = len(set(classifier_list).intersection(configs['model_hierarchy']['deep_learning'])) > 0

        return all([has_tree_based, has_lstm])
