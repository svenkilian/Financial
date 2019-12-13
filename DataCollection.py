"""
This module implements methods to collect financial data from Wharton Research Services via the wrds package
"""

# Imports
import json
import os
import sys
import time
import wrds
import re
import pandas as pd
import datetime
import numpy as np

from core import data_processor
from utils import plot_data, pretty_print

# Configurations for displaying DataFrames
pd.set_option('precision', 3)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 15)
pd.set_option('max_colwidth', 25)
pd.set_option('mode.sim_interactive', True)
pd.set_option('expand_frame_repr', True)
pd.set_option('large_repr', 'truncate')

pd.set_option('colheader_justify', 'left')
pd.set_option('display.width', 800)
pd.set_option('display.html.table_schema', False)


def get_data_table(db: wrds.Connection, sql_query=False, query_string='', library=None, table=None, columns=None,
                   obs=-1,
                   index_col=None, date_cols=None, recent=False, n_recent=100, params=None,
                   table_info=1) -> pd.DataFrame:
    """
    Get pandas DataFrame containing the queries data from the specified library and table.

    :param db: wrds database connection
    :param sql_query: SQL query flag
    :param query_string: SQL query string
    :param library: Library to use
    :param table: Table withing library
    :param columns: Columns to include in the DataFrame
    :param obs: Number of observations
    :param index_col: Index of index columns
    :param date_cols: Index of data columns
    :param recent: Flag indicating whether to only return recent data
    :param n_recent: Number of recent data rows to return
    :param table_info: Integer indicating info verbosity
    :param params: Addition SQL query parameter
    :return: DataFrame containing the queried table
    """

    print('Processing query ...')

    # JOB: Extract library and table name from SQL query string
    if sql_query and query_string:
        lib_tab_string = re.search('(?:from|FROM)\s(\S*)', query_string).group(1)
        library, table = lib_tab_string.split('.')

    # JOB: Print information about the queried table
    if table_info == 1:
        print('Approximate number of rows in %s.%s: %d' % (library, table, db.get_row_count(library, table)))
    elif table_info == 2:
        print('Queried table information for %s - %s: ' % (library, table))
        print(db.describe_table(library, table))

    if not sql_query:
        # Standard non-SQL query
        data_table = db.get_table(library=library, table=table, columns=columns, obs=obs, index_col=index_col,
                                  date_cols=date_cols)

    else:
        # Query via SQL string
        data_table = db.raw_sql(query_string, index_col=index_col, date_cols=date_cols, params=params)

    if recent:
        # In case only n_recent dates are queried

        # Create array of last n date indices (deduplicated)
        last_n_indices = data_table.set_index(keys='datadate').index.to_frame().iloc[:, 0].sort_values(
            ascending=True).drop_duplicates().values[
                         -n_recent:]

        data_table.set_index(keys='datadate', inplace=True)

        # Query last n dates from DataFrame
        data_table = data_table.loc[last_n_indices, :]

    return data_table


def list_tables(db, library, show_n_rows=False):
    """
    List all tables in a given library with optional row count information
    :param db: Database connection
    :param library: Queries library
    :param show_n_rows: Show number of rows
    :return:

    Usage::
            >>> list_tables(db, 'library_name', show_n_rows=False)
            bank_ifndytd
            bank_ifntq
            ...
    """
    for table in db.list_tables(library):
        if show_n_rows:
            print(table + ': %d rows' % db.get_row_count(library, table))
        else:
            print(table)


def get_index_constituents(constituency_matrix: pd.DataFrame, date: datetime.date) -> list:
    """
    Return company name list of index constituents for given date

    :param lookup_dict: Dictionary mapping gvkey to company name
    :param constituency_matrix: Constituency table providing constituency information
    :param date: Date for which to return constituency list

    :return: List of company names for given date
    :rtype: list
    """

    lookup_dict = pd.read_json(os.path.join('data', 'gvkey_name_dict.json'), typ='series').to_dict().get('conm')
    # print(lookup_dict)
    # print(len(constituency_matrix.loc[date].loc[lambda x: x == 1]))
    # print(constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values('gvkey').tolist())
    constituent_list_names = [lookup_dict.get(key) for key in
                              constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values(
                                  'gvkey').tolist()]

    return constituency_matrix.loc[date].loc[lambda x: x == 1].index


def download_index_history(index_id: str, from_file=False, last_n=None):
    """
    Download complete daily index history

    :return:
    :rtype:
    """

    if not from_file:
        gvkeyx_lookup = pd.read_csv(os.path.join('data', 'Compustat_Global_Indexes.csv'), index_col='GVKEYX')
        gvkeyx_lookup.index = gvkeyx_lookup.index.astype(str)
        gvkeyx_lookup = gvkeyx_lookup['index_name'].to_dict()

        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')

        gvkey_list, relevant_date_range = create_constituency_matrix(load_from_file=True)
        if last_n:
            start_date = str(relevant_date_range[-1].date() - datetime.timedelta(days=last_n - 1))
        else:
            start_date = str(relevant_date_range[0].date())

        end_date = str(relevant_date_range[-1].date())

        # Specify list of companies and start and end date of query
        parameters = {'company_codes': tuple(gvkey_list), 'start_date': start_date, 'end_date': end_date}

        print('Querying full index history for index %s \n'
              'between %s and %s ...' % (gvkeyx_lookup.get(index_id), start_date, end_date))
        start_time = time.time()
        data = get_data_table(db, sql_query=True,
                              query_string="select datadate, gvkey, iid, trfd, ajexdi, cshtrd, prccd, divd, conm, curcdd, sedol, exchg "
                                           "from comp.g_secd "
                                           "where gvkey in %(company_codes)s and datadate between %(start_date)s and %(end_date)s "
                                           "order by datadate asc",
                              index_col=['datadate', 'gvkey', 'iid'], table_info=1, params=parameters)

        end_time = time.time()

        print('Query duration: %g seconds' % (round(end_time - start_time, 2)))

        # JOB: Calculate Return Index Column
        data['return_index'] = (data['prccd'] / data['ajexdi']) * data['trfd']

        # JOB: Calculate Daily Return
        data['daily_return'] = data.groupby(level=['gvkey', 'iid'])['return_index'].apply(
            lambda x: x.pct_change(periods=1))

        # Save to file
        data.to_csv(os.path.join('data', 'index_data_constituents.csv'))

    else:
        data = pd.read_csv(os.path.join('data', 'index_data_constituents.csv'), dtype={'gvkey': str})
        print(data.index.drop_duplicates)
        data.set_index('gvkey', inplace=True)

    # JOB: Create company name dictionary and save to file
    company_name_lookup = data.reset_index()
    company_name_lookup = company_name_lookup.loc[
        ~company_name_lookup.duplicated(subset='gvkey', keep='first'), ['gvkey', 'conm']].set_index('gvkey')
    company_name_lookup.to_json(os.path.join('data', 'gvkey_name_dict.json'))


def create_constituency_matrix(load_from_file=False, index_id='150095') -> tuple:
    """
    Generate constituency matrix for stock market index components

    :param index_id: Index to create constituency matrix for
    :param load_from_file: Flag indicating whether to load constituency information from file

    :return: Tuple containing [0] list of historical constituent gvkeys and [1] relevant date range
    """

    # File name with constituents and corresponding time frames
    file_name = 'data_constituents.csv'
    db = None
    parameters = {'index_id': (index_id,)}

    # JOB: In case constituents have to be downloaded from database
    if not load_from_file:
        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')
        const_data = get_data_table(db, sql_query=True,
                                    query_string="select * "
                                                 "from comp.g_idxcst_his "
                                                 "where gvkeyx in %(index_id)s ",
                                    index_col=['gvkey', 'iid'], table_info=1, params=parameters)

        # Save to file
        const_data.to_csv(os.path.join('data', 'data_constituents.csv'))

    # JOB: Load table from local file
    else:
        # Load constituency table from file and transform key to string
        const_data = pd.read_csv(os.path.join('data', file_name), dtype={'gvkey': str})
        # const_data['gvkey'] = const_data['gvkey'].astype('str')

        # Convert date columns to datetime format
        for col in ['from', 'thru']:
            const_data[col] = pd.to_datetime(const_data[col], format='%Y-%m-%d')
            const_data[col] = const_data[col].dt.date

        # Set gvkey and iid as MultiIndex
        const_data.set_index(['gvkey', 'iid'], inplace=True)

    # Query relevant date range
    index_starting_date = const_data['from'].min()
    relevant_date_range = pd.date_range(index_starting_date, datetime.date.today(), freq='D')

    # Create empty constituency matrix
    constituency_matrix = pd.DataFrame(0, index=relevant_date_range,
                                       columns=const_data.index.drop_duplicates())

    # JOB: Iterate through all company stocks ever listed in index and set adjacency to 0 or 1
    for stock_index in const_data.index:
        for row in const_data.loc[stock_index].iterrows():
            if pd.isnull(row[1]['thru']):
                constituency_matrix.loc[pd.date_range(start=row[1]['from'], end=datetime.date.today()), stock_index] = 1
            else:
                constituency_matrix.loc[pd.date_range(start=row[1]['from'], end=row[1]['thru']), stock_index] = 1

    # Save constituency table to file
    constituency_matrix.to_csv(os.path.join('data', 'constituency_matrix.csv'))

    if not load_from_file:
        db.close()
        print('DB connection closed.')

    # Return historical constituents as list
    return const_data.index.get_level_values('gvkey').drop_duplicates().to_list(), relevant_date_range


def generate_study_period(constituency_matrix: pd.DataFrame, full_data: pd.DataFrame, period_range: tuple) -> pd.DataFrame:
    """
    Generate a time-period sample for a study period

    :param period_range: Date range of study period
    :type period_range: pd.DatetimeIndex
    :param full_data: Full stock data
    :type full_data: pd.DataFrame
    :param constituency_matrix: Constituency matrix
    :type constituency_matrix: pd.DataFrame

    :return: Study period sample
    :rtype: pd.DataFrame
    """

    # Convert date columns to DatetimeIndex
    full_data['datadate'] = pd.to_datetime(full_data['datadate'])

    # Get list of constituents for specified date
    full_data.set_index('datadate', inplace=True)
    unique_dates = full_data.index.drop_duplicates()
    constituent_indices = get_index_constituents(constituency_matrix, unique_dates[period_range[1]])
    full_data.reset_index(inplace=True)

    print('Retrieving index constituency for %s' % unique_dates[period_range[1]])

    # Select relevant data
    full_data = full_data.set_index(['gvkey', 'iid'])
    full_data = full_data.loc[constituent_indices, :]
    full_data = full_data.reset_index()
    full_data.set_index('datadate', inplace=True)

    # Select data from study period
    print('Retrieving data from %s to %s' % (unique_dates[period_range[0]].date(), unique_dates[period_range[1]].date()))
    study_data = full_data.loc[unique_dates[period_range[0]:period_range[1]]]

    # Add standardized daily returns
    mean_daily_return = study_data.loc[unique_dates[period_range[0]:period_range[1]], 'daily_return'].mean()
    std_daily_return = study_data.loc[unique_dates[period_range[0]:period_range[1]], 'daily_return'].std()
    print('Mean daily return: %g' % mean_daily_return)
    print('Std. daily return: %g' % std_daily_return)

    study_data['stand_d_return'] = (study_data['daily_return'] - mean_daily_return) / std_daily_return

    return study_data



def main():
    # Load constituency matrix
    constituency_matrix = pd.read_csv(os.path.join('data', 'constituency_matrix.csv'), index_col=0, header=[0, 1],
                                      parse_dates=True)

    # Load full data
    full_data = pd.read_csv(os.path.join('data', 'index_data_constituents.csv'), dtype={'gvkey': str})

    start_index = -50
    end_index = -1
    period_range = (start_index, end_index)

    study_period_data = generate_study_period(constituency_matrix=constituency_matrix, full_data=full_data,
                                              period_range=period_range)

    print(study_period_data)

    sys.exit(0)

    # JOB: Run settings:
    download_data = True
    pivot_transform = False
    data = None

    # Establish database connection
    print('Opening DB connection ...')
    db = wrds.Connection(wrds_username='afecker')
    print('Done')

    # print(db.describe_table('crspa', 'dsf'))
    # print(db.describe_table('comp', 'g_idxcst_his'))
    #
    # sys.exit(0)

    # print(list_tables(db, 'comp'))

    # indices = {'150007': 'DAX', '150008': 'FTSE 100', '150069': 'Nikkei 225 Index',
    #            '150940': 'Dow Jones Euro STOXX 50 Index', '150919': 'S&P Global 100 Index'}

    # JOB: Download data
    if download_data:
        # parameters = {'indices': tuple(indices.keys())}
        # data = get_data_table(db, sql_query=True,
        #                       query_string="select datadate, gvkeyx, prccm "
        #                                    "from comp.g_idx_mth "
        #                                    "where gvkeyx in %(indices)s and datadate between '1990-01-01' and '2019-11-01' "
        #                                    "order by datadate asc",
        #                       index_col=['datadate', 'gvkeyx'], table_info=1, params=parameters)

        data = get_data_table(db, sql_query=True,
                              query_string="select b.gvkeyx, a.gvkey, a.isin, b.from, b.thru, a.datadate, a.conm, a.cshtrd, a.prccd, a.divd, a.curcdd, a.exchg, a.fic, a.gind, a.iid, a.secstat, a.trfd "
                                           "from comp.g_secd a join comp.g_idxcst_his b on a.gvkey = b.gvkey "
                                           "where b.gvkeyx = '150095' and b.thru is null and a.isin is not null and a.datadate between '2018-01-01' and '2019-11-26' "
                                           "order by a.datadate asc",
                              index_col=['datadate', 'gvkey'], table_info=1)

        # parameters = {'indices': tuple(indices.keys())}
        # data = get_data_table(db, sql_query=True,
        #                       query_string="select date, djct "
        #                                    "from djones.djdaily "
        #                                    "where date between '1992-01-01' and '2019-11-01' "
        #                                    "order by date asc",
        #                       index_col=['date'], columns='djct', table_info=1)

        # dow_data = get_data_table(db=db, library='djones', table='djdaily', columns=['date', 'dji'], obs=-1,
        #                           index_col='date', sql_query=False, recent=True, n_recent=10)

        data.to_csv('data/data.csv')
        # Export data to json file
        data.reset_index(inplace=True)
        data.to_json(os.path.join('data', 'data_export.json'))

    if pivot_transform:
        # JOB: Load data
        data = pd.read_csv('data/dax_data.csv', index_col=[0, 1])

        # JOB: Query first value (in a temporal sense) of each security and normalize data by respective first values
        # firsts = data.groupby(level='gvkeyx').transform('first')
        # data = data / firsts

        # JOB: Create pivot table from data
        data.reset_index(inplace=True)
        data = data.pivot_table(values=['prccd'], index=['datadate'], columns=['gvkey'], )
        # data = data.pivot_table(values=['prccm'], index=['datadate'], columns=['gvkeyx'], )

        # JOB: Drop top level of column index
        # data = data.droplevel(0, axis=1)

        # JOB: Select time range and gvxkeys
        # data = data.loc['1994-01-01':'2019-11-09', [150007]]  # [150007, 150008, 150069]]

        # JOB: Show data table head and indices
        # print(data.head(10))
        # print(data.index)
        # print(data.columns)

    # Import data file
    print('Loading data from csv file ...')
    data = pd.read_csv(os.path.join('data', 'data.csv'))
    data['datadate'] = pd.to_datetime(data['datadate'], format='%Y-%m-%d')

    # Set Multiindex
    data.set_index(keys=['datadate', 'gvkey'], inplace=True)

    data['exchg'] = data['exchg'].astype(int)
    # Filter by ISIN and exchg code
    # data = data[(data['isin'].str.startswith('DE')) & (data['exchg'] == 171)]
    data = data[data['isin'].str.startswith('DE')]

    print('Data loaded successfully.')

    # print(data.index)
    # print(data.columns)

    # Filter out duplicates
    data = data[~data.index.duplicated(keep='first')]

    print(len(data.loc['2019-11-18':'2019-11-18']))
    print(data.loc['2019-11-18':'2019-11-18'])

    # JOB: Initialize DataLoader
    # data_l = data_processor.DataLoader(data, split=0.75, cols=data.columns.tolist())
    # Generate training data
    # train_x, train_y = data_l.get_train_data(
    #     seq_len=10, normalize=True)

    # Generate test data
    # test_x, test_y = data_l.get_test_data(seq_len=10, normalize=True)

    # print('Training Data:')
    # print('Features')
    # print(train_x[0:5])
    # print('\nLabels:')
    # print(train_y[:5])
    #
    # print('\n\nTest Data:')
    # print('Features')
    # print(test_x[5:15])
    # print('\nLabels:')
    # print(test_y[5:15])

    # plot_data(data.loc[:], columns=data.columns, index_name=None, title='Dow Jones Data',
    #           col_multi_index=False,
    #           label_mapping=None, export_as_png=True)

    # Close database connection
    db.close()
    print('DB connection closed.')


# Main method
if __name__ == '__main__':
    main()
    # create_constituency_matrix(load_from_file=True)
    # download_index_history(index_id='150095', from_file=False, last_n=100)

# -------------------------------
# Plotting multiple measures for a single security
# -------------------------------

# # JOB: Melt measures into long form
# data.reset_index(inplace=True)
# data = pd.melt(data, id_vars=['datadate', 'conm'], value_vars=['cshtrd', 'prccd', 'cshoc'], var_name='measure')
#
# # JOB: Create pivot table
# data_pivot = data.pivot_table(values='value', index=['datadate', 'measure'], columns=['conm'])
# # data_pivot.index = pd.to_datetime(data_pivot.index)
# print(data_pivot)
#
# # print(data_pivot.loc[(slice(None), 'cshtrd'), :])
#
# plot_data(data_pivot.loc[(slice('2019-01-01', '2019-11-01'), 'cshtrd'), :], columns=data_pivot.columns,
#           title='Trading Volume')
# plt.show()


# ---------------------------------
# JOB: Example queries
# ---------------------------------

# JOB: Get DAX constituents for specific period
# data = get_data_table(db, sql_query=True,
#                       query_string="select b.gvkeyx, a.gvkey, a.isin, b.from, b.thru, a.datadate, a.conm, a.cshtrd, a.prccd, a.divd, a.curcdd, a.exchg, a.fic, a.gind, a.iid, a.secstat, a.trfd "
#                                    "from comp.g_secd a join comp.g_idxcst_his b on a.gvkey = b.gvkey "
#                                    "where b.gvkeyx = '150007' and b.thru is null and a.isin is not null and a.datadate between '2017-01-01' and '2019-11-26' "
#                                    "order by a.datadate asc",
#                       index_col=['datadate', 'gvkey'], table_info=2)

# JOB: Get select current DAX members and return various key figures
# company_codes = ['015575', '015576', '015677']
# parameters = {'company_codes': tuple(company_codes)}
# data = get_data_table(db, sql_query=True,
#                       query_string="select b.gvkeyx, a.gvkey, a.isin, b.from, b.thru, a.datadate, a.conm, a.cshtrd, a.prccd, a.divd, a.cshoc "
#                                    "from comp.g_secd a join comp.g_idxcst_his b on a.gvkey = b.gvkey "
#                                    "where b.gvkeyx = '150007' and a.gvkey in %(company_codes)s and b.thru IS NULL and a.cshoc IS NOT NULL and a.isin IS NOT NULL and a.datadate between '2010-11-01' and '2019-11-01' "
#                                    "order by a.datadate asc",
#                       index_col=['datadate', 'gvkey'], params=parameters, table_info=1)

# JOB: [Non-SQL] Get Dow Jones daily index data
# dow_data = get_data_table(db=db, library='djones', table='djdaily', columns=['date', 'dji'], obs=-1,
#                           index_col='date', sql_query=False, recent=True, n_recent=10)

# JOB: [SQL] Get Dow Jones daily index data
# dow_data_sql = get_data_table(db, sql_query=True, query_string='select date,dji from djones.djdaily LIMIT 10;',
#                               date_cols='date', recent=True, n_recent=100)

# data = get_data_table(db, sql_query=True,
#                       query_string="select cusip, permno, date, bidlo, askhi "
#                                    "from crsp.dsf "
#                                    "where permno in (14593, 90319, 12490, 17778) and "
#                                    "date between '2010-01-01' and '2013-12-31' and "
#                                    "askhi > 2000",
#                       date_cols='date')


# JOB: Query data for IBM from joining fundamenal table with monthly data
# db.raw_sql("select a.gvkey, a.datadate, a.tic, a.conm, a.at, a.lt, b.prccm, b.cshoq "
#            "from comp.funda a join comp.secm b on a.gvkey = b.gvkey and a.datadate = b.datadate "
#            "where a.tic = 'IBM' and a.datafmt = 'STD' and a.consol = 'C' and a.indfmt = 'INDL'")


# JOB: Query company data in certain time frame
# values = ','.join(['datadate', 'conm', 'gvkey', 'prcod', 'prcld', 'prchd'])
# company_keys = ('001491')
# # parm = {'values': values, 'company_keys': tuple(company_keys)}
# data = get_data_table(db, sql_query=True,
#                       query_string="select %(values)s "
#                                    "from comp.g_secd "
#                                    "where datadate between '2019-01-01' and '2019-03-01' "
#                                    # "and gvkey in %(company_keys)s "
#                                    "and gvkey = '001491' "
#                                    "order by datadate "
#                                    "asc " % {'values': values, 'company_keys': company_keys},
#                       index_col=['datadate', 'gvkey'], table_info=1)
