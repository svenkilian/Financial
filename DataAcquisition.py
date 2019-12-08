"""
This module provides functions to download sample data sets from .csv files as well as sample
LSTM model files provided in GitHub repositories
"""

import requests
import csv
import pandas as pd


def download_data():
    file_url = 'https://raw.githubusercontent.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/master/data/sp500.csv'
    data = pd.read_csv(file_url)
    data.to_csv('data/' + file_url.split('/')[-1], index=False)


def download_py_file():
    # file_url = 'https://raw.githubusercontent.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/master/run.py'
    file_url = 'https://raw.githubusercontent.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/master/config.json'
    response = requests.get(file_url)
    print(response.headers['Content-Type'])
    print(file_url)

    with open(file_url.split('/')[-1], 'wb') as file:
        file.write(response.content)


if __name__ == '__main__':
    download_py_file()
