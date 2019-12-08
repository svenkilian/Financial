__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import LSTMModel
from keras import metrics
import utils


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


def main(load_latest_model=False):
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])

    # JOB: Initialize DataLoader
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        # pd.read_json(os.path.join('data', 'data_export.json'), orient='split'),
        configs['data']['train_test_split'],
        configs['data']['columns'], from_csv=True
    )

    # JOB: Build model
    model = LSTMModel()
    model.build_model(configs)

    # JOB: Generate training data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalize=configs['data']['normalize']
    )

    # JOB: In-memory training
    history = model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    """
    # Out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
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

    # JOB: Generate test data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalize=configs['data']['normalize']
    )

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
    #                                                configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # JOB: Make point prediction
    predictions = model.predict_point_by_point(x_test)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # JOB: Plot results
    plot_results(predictions, y_test)

    # JOB: Plot training and validation metrics
    utils.plot_train_val(history)

    test_scores = model.model.evaluate(x_test, y_test, verbose=0)

    print(pd.DataFrame(test_scores, index=model.model.metrics_names).T)


if __name__ == '__main__':
    main(load_latest_model=True)
