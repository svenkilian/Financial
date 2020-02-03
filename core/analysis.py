"""
This module implements classes and methods for data analysis
"""
import itertools
import json
import webbrowser
from textwrap import wrap
from typing import Union

import matplotlib.dates as mdates
import numpy as np
from colorama import Fore, Style
from matplotlib import ticker
from matplotlib.pyplot import gca
from tqdm import tqdm

from config import *
from core.data_collection import add_constituency_col, to_actual_index, load_full_data
from core.utils import pretty_print_table, Timer
from external.dm_test import dm_test


class StatsReport:
    """
    Implement StatsReport object, which represents a wrapped DataFrame loaded from a json file
    """

    def __init__(self, log_path: str = None, attrs: list = None, model_types=None):
        """

        :param attrs: (Optional) Provide list of columns else - if None, infer from headers
        """

        if log_path is None:
            log_path = os.path.join(ROOT_DIR, 'data', 'training_log.json')
        else:
            log_path = os.path.join(ROOT_DIR, 'data', log_path)

        try:
            print(f'Log path: {log_path}')
            self.data = pd.read_json(log_path, orient='index',
                                     convert_dates=['Experiment Run End', 'Test Set Start Date', 'Test Set End Date',
                                                    'Study Period Start Date', 'Study Period End Date'])
        except ValueError as ve:
            print('Log file does not exist.')
            print(ve)
            exit(1)

        if attrs is None:
            self.attrs = self.data.columns
        else:
            self.attrs = attrs

        if model_types is not None:
            self.data = self.data[self.data['Model Type'].isin(model_types)]

    def to_html(self, title='Statistics', file_name='stats', open_window=True):
        """
        Create and save HTML table

        :param title: Title of HTML page
        :param file_name: File name to save HTML page to
        :param open_window: Open created file in new browser window
        :return:
        """
        data = self.split_dict_cols()
        table = data.to_html(classes='blueTable')
        html_doc = f"""
        <html>
            <head><title>{title}</title></head>
            <link rel="stylesheet" type="text/css" media="screen" href="tbl_style.css" />
            <body>
                {table}
            </body>
        </html>
        """
        with open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), 'w') as f:
            f.write(html_doc)

        if open_window:
            webbrowser.open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), new=2)

    def summary(self, score_list: list = None, index_only: str = None, k: Union[list, int] = None, last_only=True,
                show_all=False,
                by_model_type=False, sort_by: str = None, run_id=None,
                show_std=False, to_html=True, open_window=True, pretty_print=False, compare_errors=False,
                start_date=None,
                end_date=None, silent=False) -> pd.DataFrame:
        """
        Print summary of StatsReport and return as DataFrame

        :param silent:
        :param pretty_print:
        :param end_date:
        :param start_date:
        :param compare_errors:
        :param run_id:
        :param show_all: Show full training history
        :param to_html: Export summary table to HTML file
        :param open_window: Open created HTML page in new browser window
        :param index_only: Filter by specified index
        :param show_std: Show standard deviation along with mean
        :param sort_by: Column to sort by
        :param by_model_type: Show results separately for each model
        :param k: List of k Top-k statistics to consider
        :param score_list: List of performance metrics to include
        :param last_only: Show statistics for last run only
        :return: Summary as DataFrame
        """

        data = self.split_dict_cols(drop_configs=True)  # Split dict columns
        index_name = None
        if isinstance(k, int):
            k = [k]

        # JOB: Select relevant columns
        if score_list is not None:
            columns = [col for col in data.columns if any([score in col for score in score_list])]
            if k:
                columns = [col for col in columns for i in k if col.endswith(f'_{i}')]
        else:
            columns = data.columns

        if compare_errors:
            columns = ['Prediction Error']

        # JOB: Filter data
        data = filter_df(data, columns=data.columns, index_only=index_only, last_only=last_only, run_id=run_id,
                         last_run_id=self.last_id, start_date=start_date, end_date=end_date)

        n_models = data['Model Type'].nunique()
        if data.shape[0] > n_models and not show_all:
            if not silent:
                print(f'Showing last {n_models} out of {data.shape[0]}')
                print('...')
                pretty_print_table(data.tail(n_models).iloc[:, :10])
        else:
            pretty_print_table(data.iloc[:, :10])

        total_obs = len(data['Test Set Start Date'].unique())
        if not silent:
            print(f'\nNumber of unique time periods in filtered observation data: {total_obs}')

        # JOB: Print summary statistics for relevant columns
        if not by_model_type:
            for attr in columns:
                print(Statistics(data.get(attr), name=attr))
        else:
            df = pd.DataFrame({attr: [] for attr in columns})
            df.index.name = 'Model'
            for attr in columns:
                model_types = data['Model Type'].unique()
                for model_type in model_types:
                    model_data = data.loc[data['Model Type'] == model_type]
                    df.loc[model_type, attr] = model_data[attr].mean()
                    if show_std:
                        df.loc[model_type, f'{attr}_std'] = model_data[attr].std()

            # JOB: Sort columns lexicographically:
            df = df.reindex(sorted(df.columns), axis=1)
            sort_by = [col for col in data.columns if sort_by in col][0].split('_')[0]
            if sort_by and k:
                sort_by_column = f'{sort_by}_{k[0]}'
                if not silent:
                    print(f'Table sorted by {sort_by_column}')
                df.sort_values(by=sort_by_column, axis=0, ascending=False, inplace=True)
                col_names_rm = [col for col in df.columns if col not in sort_by_column]
                df = df[[sort_by_column, *col_names_rm]]
            else:
                df.sort_values(by=columns, axis=0, ascending=False, inplace=True)
            if to_html:
                df_to_html(df,
                           title='Model Performance Overview' if not index_only else f'Model Performance Overview {index_name}',
                           open_window=open_window, file_name='mode_performance_overview')

            if pretty_print:
                pretty_print_table(df)

            return df

    def compare_prediction_errors(self, to_html=True):
        """
        Compare prediction errors

        :return:
        """

        data = filter_df(self.data, columns=['Prediction Error'], index_only=None, last_only=True,
                         last_run_id=self.last_id)

        model_types = data['Model Type'].unique()
        model_combinations = list(itertools.permutations(model_types, 2))

        relative_performance_table = pd.DataFrame(columns=model_types, index=model_types)
        print('Creating relative performance table ...')
        timer = Timer().start()
        for combination in tqdm(model_combinations, leave=False):
            if data.loc[data['Model Type'] == combination[0], 'Prediction Error'].values[0] is None or \
                    data.loc[data['Model Type'] == combination[1], 'Prediction Error'].values[0] is None:
                continue
            error_1 = [errors for error_list in
                       data.loc[data['Model Type'] == combination[0], 'Prediction Error'].values for
                       errors in error_list]

            error_2 = [errors for error_list in
                       data.loc[data['Model Type'] == combination[1], 'Prediction Error'].values for
                       errors in error_list]

            # print(f'\nTesting H_0: {combination[0]} > {combination[1]}')
            # print(f'Total errors of {combination[0]}: {sum(error_1)}')
            # print(f'Total errors of {combination[1]}: {sum(error_2)}')
            p_value = dm_test(e_1=error_1, e_2=error_2, power=1, alternative="greater").p_value
            # print(f'p-value: {p_value}\n')
            relative_performance_table.loc[combination[1], combination[0]] = p_value

        relative_performance_table.fillna('-', inplace=True)

        if to_html:
            df_to_html(relative_performance_table,
                       title='Model Prediction Performance Comparison',
                       open_window=False, file_name='mode_prediction_comparison')

        pretty_print_table(relative_performance_table)
        timer.stop()

    # noinspection DuplicatedCode
    def plot_return_series(self, cumulative_only=True):
        """

        :param cumulative_only:
        :return:
        """
        data = filter_df(self.data, columns=['Prediction Error'], index_only=None, last_only=True,
                         last_run_id=self.last_id)

        model_types = data['Model Type'].unique()

        market_return_series = pd.Series()
        for return_period in data.loc[data['Model Type'] == 'Market', 'Return Series'].values:
            market_return_period_series = pd.Series(return_period)
            market_return_period_series.index = pd.DatetimeIndex(market_return_period_series.index)
            market_return_series = pd.concat([market_return_series, market_return_period_series])

        cumulative_market_return = (market_return_series + 1).cumprod().rename('Cumulative Market Return')
        cumulative_market_return.index.name = 'Time'
        market_return_series.index.name = 'Time'

        for model in model_types:
            if model != 'Market':
                return_series = pd.Series()
                for return_period in data.loc[data['Model Type'] == model, 'Return Series'].values:
                    return_period_series = pd.Series(return_period)
                    return_period_series.index = pd.DatetimeIndex(return_period_series.index)
                    return_series = pd.concat([return_series, return_period_series])

                cumulative_return = (return_series + 1).cumprod().rename('Cumulative Portfolio Return')
                cumulative_return.index.name = 'Time'
                return_series.index.name = 'Time'

                cumulative_returns_merged = pd.concat([cumulative_return, cumulative_market_return], axis=1,
                                                      join='outer')

                cumulative_returns_merged.plot()

                plt.title(label='\n'.join(wrap(model.replace('_', ' '), 60)), fontsize=10)
                plt.tight_layout()
                plt.legend(loc='best')
                plt.show()

                if not cumulative_only:
                    returns_merged = pd.concat(
                        [return_series.rename('Excess Portfolio Return'), market_return_series.rename(
                            'Excess Market Return')], axis=1, join='outer')
                    returns_merged = returns_merged.resample('Q').mean()
                    returns_merged.loc['2005-08':'2019-11', :].plot()
                    plt.title(label='\n'.join(wrap(model.replace('_', ' '), 60)), fontsize=10)
                    plt.tight_layout()
                    plt.legend(loc='best')
                    plt.show()

    def split_dict_cols(self, drop_configs=True):
        """
        Split columns containing dicts into separate columns

        :return: DataFrame with split columns
        """
        data = self.data.copy().get(self.attrs)
        if drop_configs:
            data.drop(columns=['Model Configs', 'Prediction Error', 'Return Series'], inplace=True)
        for attr in data:
            if self.data[attr].dtype == object:
                if isinstance(data[attr][-1], dict) and attr != 'Model Configs':
                    # Split up dict entries into separate columns
                    for key in data[attr][-1].keys():
                        data[f'{attr}_{key}'] = data[attr].apply(
                            lambda x: x if (isinstance(x, float) or x is None) else x.get(key))

                    # Drop original columns
                    data.drop(labels=attr, inplace=True, axis=1)
        return data

    @property
    def last_id(self):
        return int(self.data.get('ID').max())


class Statistics:
    """
    Class implementing a statistic
    """

    def __init__(self, data=None, index=None, name=None):
        """
        Constructor for Statistics class

        :param data: Series-like data object
        :param index: (Optional) Index for data series
        :param name: Name of data series
        """
        self.name = name
        if hasattr(data, 'index'):
            index = data.index
        self.data = pd.Series(data, name=self.name, index=index) if data is not None else None

    def __repr__(self):
        summary = None
        if self.data is not None:
            summary = f'{30 * "-"}\n' \
                      f'Name: {self.name}\n' \
                      f'Count: {self.data.count}\n' \
                      f'Unique: {self.nunique}\n' \
                      f'Mean: {np.round(self.mean, 4)}\n' \
                      f'Stdv.: {np.round(self.std, 4)}\n' \
                      f'{30 * "-"}\n'
        return summary

    @property
    def count(self):
        return pd.Series.count(self.data)

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def std(self):
        return np.std(self.data)

    @property
    def nunique(self):
        return self.data.nunique()


class VisTable:
    """
    Implements a visualization table from a DataFrame with functions for plotting
    """

    def __init__(self, data: pd.DataFrame, time_frame: tuple = None, groups: list = None, freq='M',
                 stat: Union[str, None] = 'mean',
                 attrs: list = None):
        self.data = data.copy()
        self.time_frame = time_frame
        self.groups = groups
        self.attrs = attrs
        self.freq = freq
        self.stat = stat

    def __repr__(self):
        return self.plot_grouped()

    @property
    def grouped_data(self):
        if self.data.index.name is not 'datadate':
            self.data.loc[:, 'Test Set End Date'] = pd.to_datetime(self.data['Test Set End Date'])
            df = self.data[(self.data['Test Set End Date'] > self.time_frame[0]) & (
                    self.data['Test Set End Date'] <= self.time_frame[1])].groupby(
                [pd.Grouper(key='Test Set End Date', freq='Y'), *self.groups])
        else:
            df = self.data.loc[self.time_frame[0]:self.time_frame[1]].groupby(self.groups)

        return df

    def plot_bar(self, title=None, legend=True, save_to_file=False):
        """
        Plot grouped bar plot over time

        :param model_types:
        :param attrs:
        :param title:
        :param legend:
        :param save_to_file:
        :return:
        """

        tf = [True, False]
        for legend in tf:
            title = self.attrs[0].replace('_', ' ')
            fig, ax = plt.subplots(figsize=(10, 6))

            data = self.grouped_data.mean()[self.attrs].unstack().droplevel(level=0, axis=1)
            data.index = data.index.year
            data.plot(kind='bar', rot=0, ax=ax, fontsize=14, width=0.8, legend=legend)
            ax.set_xlabel('Year')
            ax.set_ylabel(title)
            ax.set_title(title)

            if 'Accuracy' in self.attrs[0]:
                ax.set_ylim(0.48, 0.58)

            plt.tight_layout()

            suffix = '_legend' if legend else ''

            file_name = '{}__{}_{}{}.png'.format(title, self.time_frame[0].replace('\'', ''),
                                                 self.time_frame[1].replace('\'', ''), suffix)
            # JOB: Save figure to file
            plt.savefig(os.path.join(ROOT_DIR, 'data/plots', file_name), dpi=600, facecolor='w', bbox_inches='tight')

            plt.show()

    def plot_grouped(self, title=None, legend=True, save_to_file=False, y_limit: float = None):
        """
        Plot data in a grouped fashion

        :param save_to_file:
        :param legend:
        :param title:
        :param y_limit:
        :return:
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        if y_limit is None:
            y_limit = 0.25

        if self.stat in ['count', 'share', 'percentage']:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, y_limit)
        if 'return' in self.attrs:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=1))

        for key, group in self.grouped_data:
            # print(key)
            if self.stat not in ['count', 'share', 'percentage']:
                ax.plot(group.groupby(pd.Grouper(freq=self.freq))[self.attrs].apply(getattr(np, self.stat)),
                        label=key)
            else:
                avg_index_len = int(round(self.data.groupby('datadate')['gvkey'].count().mean()))
                ax.plot(
                    group.groupby(pd.Grouper(freq=self.freq))[self.attrs].size().resample('Y', how='mean').divide(
                        avg_index_len),
                    label=key)

        if title is not None:
            ax.set_title(title.title(), fontsize=16)
        else:
            ax.set_title(', '.join([attr.replace('_', ' ').title() for attr in self.attrs]), fontsize=16)

        if legend:
            legend = ax.legend(loc='best', fontsize='large', shadow=False)
            for line in legend.get_lines():
                line.set_linewidth(3.0)

        if save_to_file:
            path_to_data = os.path.join(ROOT_DIR, 'data/plots')
            if not os.path.exists(path_to_data):
                print(f'Creating folder for plots ...')
                os.mkdir(path_to_data)

            plt.savefig(os.path.join(path_to_data, f'{title.lower().replace(" ", "_")}.png'), dpi=600)
        plt.show()


class DataTable:
    """
    Implements a DataFrame wrapper class with grouping and date indexing capability
    """

    def __init__(self, data: pd.DataFrame, attrs: list = None, time_frame: tuple = None, groups: list = None, freq='M'):
        self.name = None
        self.time_frame = time_frame
        self.groups_ = groups
        self.freq = freq
        self.data = data
        self.attrs = attrs

    def __repr__(self):
        return self.data.__repr__()

    @property
    def groups(self):
        return self.data.loc[self.time_frame[0]:self.time_frame[1]].groupby(self.groups_)[self.attrs]

    @property
    def combined(self):
        return self.data.loc[self.time_frame[0]:self.time_frame[1]]

    def get_group_stats(self):
        """
        Create DataFrame containing grouped stats

        :return:
        """
        summary = pd.DataFrame([], columns=['mean', 'std', 'skew', 'kurt', 'count'])
        summary.index.name = 'Industry'

        all_stats = self.combined.groupby(pd.Grouper(freq=self.freq))[
            self.attrs].mean().agg(['mean', 'std', 'skew', pd.DataFrame.kurt])
        n_all = self.combined.groupby(pd.Grouper(freq=self.freq))[self.attrs].size().mean()
        all_stats.loc['count', self.attrs] = n_all

        for key, group in self.groups:
            print(key)
            group_stats = group.groupby(pd.Grouper(freq=self.freq))[self.attrs].mean().agg(
                ['mean', 'std', 'skew', pd.DataFrame.kurt])
            n_stocks = group.groupby(pd.Grouper(freq=self.freq))[self.attrs].size().mean()
            group_stats.loc['count', self.attrs] = n_stocks

            summary.loc[key] = group_stats[self.attrs[0]]
            summary = summary[['count', 'mean', 'std', 'skew', 'kurt']]

        summary.loc['All'] = all_stats[self.attrs[0]]

        return summary


def fill_na(data: pd.DataFrame, columns: list = None):
    """
    Backfill missing values

    :param columns: Columns to backfill
    :param data:
    :return:
    """
    data = data.copy()
    data.reset_index(inplace=True)
    data.set_index(['datadate', 'gvkey', 'iid'], inplace=True)
    data.sort_index(level='datadate', inplace=True)
    data = data.unstack(0).stack(dropna=False)
    for col in columns:
        data.loc[:, f'{col}_filled'] = data.groupby(['gvkey', 'iid'])[col].bfill(limit=4)

    return data


def resample_month_end(data, columns: list = None):
    """
    Resample data to monthly frequency and take last value in month

    :param data: DataFrame
    :param columns:
    :return:
    """
    data = data.copy()
    data = fill_na(data, ['return_index'])  # Backfill missing daily values

    data.reset_index(inplace=True)
    data.set_index(['datadate', 'gvkey', 'iid'], inplace=True)
    data.sort_index(level='datadate', inplace=True)

    data = data.groupby(['gvkey', 'iid']).resample('M', level='datadate').apply('last')  # Resample to monthly frequency
    data.dropna(subset=['return_index_filled'], inplace=True)
    # Create monthly return variable
    data.loc[:, 'monthly_return'] = data.groupby(['gvkey', 'iid'])['return_index_filled'].apply(
        lambda x: x.pct_change(periods=1))
    data = data.reorder_levels(order=['datadate', 'gvkey', 'iid'])
    data.sort_index(level='datadate', inplace=True)
    if columns:
        data = data.loc[:, ['gics_sector', *columns]]
    data.reset_index(['gvkey', 'iid'], inplace=True)

    return data


def filter_df(df, columns=None, index_only: str = None, run_id=None, last_only=True, last_run_id=None, start_date=None,
              end_date=None):
    """
    Filter data frame

    :param end_date:
    :param start_date:
    :param last_run_id:
    :param df:
    :param columns:
    :param index_only:
    :param run_id:
    :param last_id:
    :param last_only:
    :return:
    """
    data_frame = df.copy()

    # JOB: Filter by specified index
    if index_only:
        try:
            index_name = data_frame.get('Index Name').loc[
                data_frame['Index Name'].str.contains(index_only, case=False, regex=False)].drop_duplicates().values[0]
        except IndexError as ie:
            print(f'No records found for {index_only}.\n'
                  f'Resorting to showing summary statistics for all indices in selected runs.')
        else:
            print(f'Filter by index: {Style.BRIGHT}{Fore.LIGHTBLUE_EX}{index_name}{Style.RESET_ALL}')
            data_frame = data_frame.loc[data_frame['Index Name'].str.contains(index_name, case=False, regex=False)]

    # JOB: Filter by specific run id
    if run_id:
        data_frame = data_frame.loc[data_frame['ID'] == run_id]
        last_only = False

    # JOB: Filter by specific run only
    if last_only:
        print(f'Summary report for last available run (ID={last_id(data_frame)} of {last_run_id}) and \n'
              f'{columns}\n')
        try:
            data_frame = data_frame.loc[data_frame['ID'] == last_id(data_frame)]
        except TypeError as te:
            print(te)
            pass
    else:
        print(f'Summary report for all runs and \n'
              f'{columns}\n')

    # JOB: Filter by specific date range
    if start_date is not None and end_date is not None:
        data_frame = data_frame.loc[
            (data_frame['Test Set Start Date'] > start_date) & (data_frame['Test Set End Date'] <= end_date)]

    return data_frame


def df_to_html(df, title='Statistics', file_name='table', open_window=True) -> None:
    """
    Export DataFrame to HTML page

    :param df: DataFrame to export as HTML
    :param title: HTML page title
    :param file_name: File name to save HTML to
    :param open_window: Open HTML page in new browser window
    :return: None
    """
    try:
        table = df.to_html(classes='blueTable')
    except AttributeError as ae:
        print('Single statistic cannot be exported to HTML.')
        return

    html_doc = f"""
    <html>
        <head><title>{title}</title></head>
        <link rel="stylesheet" type="text/css" media="screen" href="tbl_style.css" />
        <body>
            {table}
        </body>
    </html>
    """
    with open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), 'w') as f:
        f.write(html_doc)

    if open_window:
        webbrowser.open(os.path.join(ROOT_DIR, 'data', f'{file_name}.html'), new=2)


def plot_yearly_metrics(metrics, k, top_n=3):
    """

    :param top_n:
    :param k:
    :param metrics:
    :return:
    """

    columns = None

    if metrics is not None:
        columns = [col for col in StatsReport().split_dict_cols().columns if
                   any([score in col for score in metrics])]
        if k:
            columns = [col for col in columns if (col.endswith(f'_{k}') and 'atc' not in col)]
    print(f'{Style.BRIGHT}{Fore.RED}Printing columns: {Style.RESET_ALL}')
    for metric in columns:

        model_types = [*(StatsReport().summary(last_only=True, index_only=None, show_all=False,
                                               score_list=metric,
                                               k=k, run_id=None,
                                               by_model_type=True,
                                               sort_by=metric,
                                               show_std=False,
                                               to_html=False,
                                               open_window=False, compare_errors=False,
                                               start_date='2002-10', end_date='2019-12',
                                               silent=True).index[
                         :top_n]), 'LSTM', 'RandomForestClassifier', 'ExtraTreesClassifier',
                       'GradientBoostingClassifier', 'Market']

        if 'accuracy' in metric.lower():
            model_types.remove('Market')

        print(f'{Style.BRIGHT}{Fore.RED}{metric}{Style.RESET_ALL}')
        report = StatsReport(log_path=None,
                             model_types=model_types)

        data = report.split_dict_cols(drop_configs=True)

        table = VisTable(data, time_frame=('1998-12', '2019-12'),
                         groups=['Model Type'], freq='Y',
                         stat=None,
                         attrs=[metric])

        table.plot_bar()


def last_id(data):
    try:
        last = int(data.get('ID').max())
    except ValueError:
        last = None
        print('Index not in DataFrame.')
    return last


def main(full_log=False, index_summary=True, plots_only=False, compare_models=False):
    """
    Main method of analysis.py module (to be called when module is run stand-alone)

    :param compare_models:
    :param full_log:
    :param index_summary: Whether to include summary of index
    :param plots_only: Whether to only plot index constituency and daily returns over time
                       (only considered if index_summary is True)
    :return:
    """
    configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))
    cols = ['above_cs_med', *configs['data']['columns']]

    index_dict = {
        'DJES': '150378',  # Dow Jones European STOXX Index
        'SPEURO': '150913',  # S&P Euro Index
        'EURONEXT': '150928',  # Euronext 100 Index
        'STOXX600': '150376',  # Dow Jones STOXX 600 Price Index
        'Europe350': '150927',
        'DAX': '150095'
    }

    # JOB: Specify index ID, relevant columns and study period length
    index_id = index_dict['Europe350']

    report = StatsReport()
    # JOB: Create model performance report

    plot_yearly_metrics(metrics=['Accuracy', 'Excess Return', 'Sharpe', 'Sortino', 'Daily'], k=150, top_n=3)

    exit()

    if compare_models:
        report.compare_prediction_errors(to_html=True)

    if full_log:
        report.to_html(file_name='run_overview', title='Training Log', open_window=False)

    report.summary(last_only=True, index_only=None, show_all=False,
                   score_list=['Accuracy', 'Sharpe', 'Sortino', 'Return'],
                   k=10, run_id=None,
                   by_model_type=True, sort_by='Sharpe', show_std=False, to_html=False,
                   open_window=False, pretty_print=True, compare_errors=False, start_date='2002-10', end_date='2018-12')

    # report.plot_return_series(cumulative_only=False)

    if index_summary:
        # Load full index data
        constituency_matrix, full_data, index_name, folder_path = load_full_data(index_id=index_id,
                                                                                 force_download=False,
                                                                                 last_n=None, columns=cols.copy(),
                                                                                 merge_gics=False)

        # JOB: Add constituency column and reduce to actual index constituents
        full_data = add_constituency_col(full_data, folder_path)
        full_data = to_actual_index(full_data)
        full_data.set_index('datadate', inplace=True)

        vis = VisTable(full_data, time_frame=('1998-12', '2019-12'), groups=['gics_sector_name'], freq='D',
                       stat='count',
                       attrs=['gvkey'])
        vis.plot_grouped(title='Index Composition', save_to_file=True, legend=False, y_limit=0.2)

        # vis = VisTable(full_data, time_frame=('2002-01', '2019-12'), groups=['gics_sector_name'], freq='Y',
        #                stat='mean',
        #                attrs=['daily_return'])
        # vis.plot_grouped(save_to_file=True, title='legend')

        if not plots_only:
            # JOB: Resample to monthly
            data = resample_month_end(full_data)
            dt = DataTable(data=data, attrs=['monthly_return'], time_frame=('2002-01', '2019-12'),
                           groups=['gics_sector_name'],
                           freq='M')

            stats = dt.get_group_stats()
            print(stats)

            df_to_html(stats, title='Index Stats', file_name='index_stats', open_window=False)


if __name__ == '__main__':
    main(full_log=False, index_summary=False, plots_only=True, compare_models=False)
