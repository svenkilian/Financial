import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from tensorflow.keras import optimizers

from core.utils import Timer
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMModel:
    """
    LSTM model class
    """

    def __init__(self):
        self.model = Sequential()  # Initialize Keras sequential model

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

    def build_model(self, configs):
        """
        Build model from configuration file

        :param configs:
        :return:
        """
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else 0.0
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), dropout=dropout,
                                    return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'],
                           optimizer=get_optimizer(configs['model']['optimizer'], parameters=
                           configs['model']['optimizer_params']),
                           metrics=configs['model']['metrics'])

        print(self.model.optimizer.get_config())

        self.model.summary()
        print()
        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s-b%s.h5' % (
            dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs), str(batch_size)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            # TensorBoard(log_dir=os.path.join('./logs'), histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
            #             write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
            #             embeddings_data=None, update_freq='epoch')
        ]
        history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.2,
            shuffle=True
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

        return history

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
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


def get_optimizer(optimizer_name, parameters=None):
    try:
        optimizer = optimizers.get(optimizer_name).from_config(parameters)

    except ValueError:
        print('Specified optimizer name is unknown.')
        print('Resorting to RMSprop as default.')
        optimizer = optimizers.RMSprop()

    return optimizer
