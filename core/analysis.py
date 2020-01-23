"""
This module implements classes and methods for data analysis
"""
import json
import os
import webbrowser

import numpy as np
import pandas as pd
from pandas import DataFrame

from config import ROOT_DIR


class StatsReport:
    """
    Implement StatsReport object
    """

    def __init__(self, attrs: list = None):
        self.data = pd.read_json(os.path.join(ROOT_DIR, 'data', 'training_log.json'), orient='index',
                                 convert_dates=['Experiment Run End', 'Test Set Start Date', 'Test Set End Date',
                                                'Study Period Start Date', 'Study Period End Date'])

        if attrs is None:
            self.attrs = self.data.columns
        else:
            self.attrs = attrs

    def to_html(self):
        data = self.split_dict_cols()

        text_file = open(os.path.join(ROOT_DIR, 'data', 'index.html'), 'w')
        text_file.write(data.to_html())
        text_file.close()

        webbrowser.open(os.path.join(ROOT_DIR, 'data', 'index.html'), new=2)

    def summary(self, score_list: list = None, k: int = None, last_only=True, by_model_type=False, model_type=False):
        """
        Print summary of StatsReport

        :param model_type:
        :param by_model_type:
        :param k:
        :param score_list:
        :param summary_type:
        :param last_only:
        :return:
        """

        print(self.data)
        if by_model_type:
            model_types = self.data['Model Type'].unique()

            for model_type in model_types:
                print(f'Model Type: {model_type}')
                self.summary(score_list=score_list, k=k, last_only=last_only, model_type=model_type)

            return

        data = self.split_dict_cols(drop_configs=True)  # Split dict columns

        if model_type:
            data = data.loc[data['Model Type'] == model_type]

        # JOB: Select relevant columns
        if score_list:
            columns = [col for col in data.columns if any([col.startswith(score) for score in score_list])]
            if k:
                columns = [col for col in columns if col.endswith(str(k))]
        else:
            columns = data.columns

        # JOB: Print summary info
        if not last_only:
            print(f'Summary report for all runs and \n'
                  f'{columns}\n')
        else:
            print(f'Summary report for last run (ID={self.last_id}) and \n'
                  f'{columns}\n')

        # JOB: Print summary statistics for relevant columns
        for attr in columns:
            try:
                if last_only:
                    print(Statistics(data.loc[data['ID'] == self.last_id, attr], name=attr))
                else:
                    print(Statistics(data.get(attr), name=attr))
            except TypeError as te:
                pass

    def split_dict_cols(self, drop_configs=True):
        """
        Split columns containing dicts into separate columns

        :return: DataFrame with split columns
        """
        data = self.data.copy().get(self.attrs)
        if drop_configs:
            data.drop(columns=['Model Configs'], inplace=True)
        for attr in data:
            if self.data[attr].dtype == object:
                if isinstance(data[attr][-1], dict) and attr != 'Model Configs':
                    # Split up dict entries into separate columns
                    for key in data[attr][-1].keys():
                        data[f'{attr}_{key}'] = data[attr].apply(lambda x: x.get(key))

                    # Drop original columns
                    data.drop(labels=attr, inplace=True, axis=1)
        return data

    @property
    def last_id(self):
        return self.data.get('ID').max()


class Statistics:
    """
    Class implementing a statistic
    """

    def __init__(self, data=None, index=None, name=None):
        self.name = name
        if hasattr(data, 'index'):
            index = data.index
        self.data = pd.Series(data, name=self.name, index=index) if data is not None else None

    def __repr__(self):
        summary = None
        if self.data is not None:
            summary = f'{30 * "-"}\n' \
                      f'Name: {self.name}\n' \
                      f'Count: {self.data.count()}\n' \
                      f'Unique: {self.data.nunique()}\n' \
                      f'Mean: {np.round(self.mean, 4)}\n' \
                      f'Stdv.: {np.round(self.std, 4)}\n' \
                      f'{30 * "-"}\n'
        return summary

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def std(self):
        return np.std(self.data)

    @property
    def nunique(self):
        return self.data.nunique()


class DataTable(DataFrame):
    _metadata = ['name']

    def __init__(self, *args, **kwargs):
        self.name = None
        self.__dict__.update(kwargs)
        rm_keys = self.__dict__.keys()

        for key in rm_keys:
            kwargs.pop(key, None)

        super(DataTable, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return DataTable
