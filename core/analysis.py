"""
This module implements classes and methods for data analysis
"""
import json
import webbrowser
from typing import Union

import numpy as np
from colorama import Fore, Style
from matplotlib import ticker
import pylab

from config import *
from core.data_collection import add_constituency_col, to_actual_index, load_full_data
from core.utils import pretty_print_table


class StatsReport:
    """
    Implement StatsReport object, which represents a wrapped DataFrame loaded from a json file
    """

    def __init__(self, attrs: list = None):
        """

        :param attrs: (Optional) Provide list of columns else - if None, infer from headers
        """
        try:
            self.data = pd.read_json(os.path.join(ROOT_DIR, 'data', 'training_log.json'), orient='index',
                                     convert_dates=['Experiment Run End', 'Test Set Start Date', 'Test Set End Date',
                                                    'Study Period Start Date', 'Study Period End Date'])
        except ValueError as ve:
            print('Log file does not exist.')
            exit(1)

        if attrs is None:
            self.attrs = self.data.columns
        else:
            self.attrs = attrs

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
                show_std=False, to_html=True, open_window=True) -> pd.DataFrame:
        """
        Print summary of StatsReport and return as DataFrame

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
        if score_list:
            columns = [col for col in data.columns if any([score in col for score in score_list])]
            if k:
                columns = [col for col in columns for i in k if col.endswith(f'_{i}')]
        else:
            columns = data.columns

        if index_only:
            # Filter by specified index
            index_name = data.get('Index Name').loc[
                data['Index Name'].str.contains(index_only, case=False, regex=False)].drop_duplicates().values[0]

            print(f'Filter by index: {Style.BRIGHT}{Fore.LIGHTBLUE_EX}{index_name}{Style.RESET_ALL}')
            data = data.loc[data['Index Name'].str.contains(index_name, case=False, regex=False)]

        # JOB: Print summary info
        if run_id:
            data = data.loc[data['ID'] == run_id]
            last_only = False

        if last_only:
            # Filter by last available index
            print(f'Summary report for last available run (ID={last_id(data)} of {self.last_id}) and \n'
                  f'{columns}\n')
            try:
                data = data.loc[data['ID'] == last_id(data)]
            except TypeError as te:
                print(te)
                pass
        else:
            print(f'Summary report for all runs and \n'
                  f'{columns}\n')

        n_models = data['Model Type'].nunique()
        if data.shape[0] > n_models and not show_all:
            print(f'Showing last {n_models} out of {data.shape[0]}')
            print('...')
            pretty_print_table(data.tail(n_models).iloc[:, :15])
        else:
            pretty_print_table(data.iloc[:, :15])

        total_obs = len(data['Test Set Start Date'].unique())
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
                print(f'Table sorted by {sort_by_column}')
                df.sort_values(by=sort_by_column, axis=0, ascending=False, inplace=True)
                col_names_rm = [col for col in df.columns if col not in sort_by_column]
                df = df[[sort_by_column, *col_names_rm]]
            else:
                df.sort_values(by=columns, axis=0, ascending=False, inplace=True)
            pretty_print_table(df)
            if to_html:
                df_to_html(df,
                           title='Model Performance Overview' if not index_only else f'Model Performance Overview {index_name}',
                           open_window=open_window, file_name='mode_performance_overview')

            return df

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
                        data[f'{attr}_{key}'] = data[attr].apply(lambda x: x.get(key))

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

    def __init__(self, data: pd.DataFrame, time_frame: tuple = None, groups: list = None, freq='M', stat='mean',
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
        return self.data.loc[self.time_frame[0]:self.time_frame[1]].groupby(self.groups)

    def plot_grouped(self, monthly=False, title=None, legend=True, save_to_file=False):
        """
        Plot data in a grouped fashion

        :return:
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        # self.data.index = pd.to_datetime(self.data.index)
        # print(self.grouped_data.get_group('Financials'))

        if self.stat in ['count', 'share', 'percentage']:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, 0.25)
        if 'return' in self.attrs:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=1))

        for key, group in self.grouped_data:
            # print(key)
            if not monthly:
                if self.stat not in ['count', 'share', 'percentage']:
                    ax.plot(group.groupby(pd.Grouper(freq=self.freq))[self.attrs].apply(getattr(np, self.stat)),
                            label=key)
                else:
                    avg_index_len = int(round(self.data.groupby('datadate')['gvkey'].count().mean()))
                    ax.plot(
                        group.groupby(pd.Grouper(freq=self.freq))[self.attrs].size().resample('Y', how='mean').divide(
                            avg_index_len),
                        label=key)
            else:
                pass

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


def last_id(data):
    try:
        last = int(data.get('ID').max())
    except ValueError:
        last = None
        print('Index not in DataFrame.')
    return last


def main(full_log=False, index_summary=True, plots_only=False):
    """
    Main method of analysis.py module (to be called when module is run stand-alone)

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
        'DAX': '150095'
    }

    # JOB: Specify index ID, relevant columns and study period length
    index_id = index_dict['DAX']

    # JOB: Create model performance report
    report = StatsReport()
    if full_log:
        report.to_html(file_name='run_overview', title='Training Log', open_window=False)
    report.summary(last_only=True, index_only='DAX', show_all=False,
                   score_list=['Accuracy', 'Sharpe', 'Sortino', 'Return'],
                   k=10, run_id=None,
                   by_model_type=True, sort_by='Annualized Excess Return', show_std=False, to_html=True,
                   open_window=True)

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

        vis = VisTable(full_data, time_frame=('2002-01', '2019-12'), groups=['gics_sector_name'], freq='D',
                       stat='count',
                       attrs=['gvkey'])
        vis.plot_grouped(title='Index Composition', save_to_file=True, legend=True)

        vis = VisTable(full_data, time_frame=('2002-01', '2019-12'), groups=['gics_sector'], freq='M',
                       stat='mean',
                       attrs=['daily_return'])
        vis.plot_grouped()

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
    main(full_log=True, index_summary=True, plots_only=True)
