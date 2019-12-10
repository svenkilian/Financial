"""
This module implements methods to collect financial data from Wharton Research Services via the wrds package
"""

# Imports
import os
import sys
import time
import wrds
import re
import pandas as pd
import pandas_datareader as data_reader

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


def get_data_table(db, sql_query=False, query_string='', library=None, table=None, columns=None, obs=-1,
                   index_col=None, date_cols=None, recent=False, n_recent=100, params=None, table_info=1):
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


def get_constituency_table(load_from_file=False):
    file_name = 'data_constituents.csv'
    # Establish database connection
    if not load_from_file:
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')
        const_data = get_data_table(db, sql_query=True,
                                    query_string="select * "
                                                 "from comp.g_idxcst_his "
                                                 "where gvkeyx = '150095' ",
                                    index_col=['gvkey'], table_info=1)

    else:
        # Load constituency table from file
        const_data = pd.read_csv(os.path.join('data', file_name))
        const_data['gvkey'] = const_data['gvkey'].astype('str')

        # Convert columns to datetime
        for col in ['from', 'thru']:
            const_data[col] = pd.to_datetime(const_data[col], format='%Y%m%d')
            const_data[col] = const_data[col].dt.date

        # Set index
        const_data.set_index(['gvkey'], inplace=True)

    # Create empty constituency table
    constituency_table = pd.DataFrame(0, index=pd.date_range('1970-01-01', '2021-01-01', freq='D'),
                                      columns=const_data.index.drop_duplicates())

    # JOB: Iterate through all companies ever listed
    for company in constituency_table.columns:
        for row in const_data[const_data.index == company].iterrows():
            if pd.isnull(row[1]['thru']):
                constituency_table.loc[pd.date_range(start=row[1]['from'], end='2021-01-01'), company] = 1
            else:
                constituency_table.loc[pd.date_range(start=row[1]['from'], end=row[1]['thru']), company] = 1

    # print(constituency_table.loc['2001-01-01':'2005-08-01'])

    # Save constituency table
    constituency_table.to_csv(os.path.join('data', 'constituency_table.csv'))

    lookup_table = const_data[~const_data.duplicated()]['co_conm'].to_dict()

    l = [lookup_table.get(key) for key in constituency_table.loc['2019-12-07'].loc[lambda x: x != 0].index.tolist()]

    pretty_print(pd.DataFrame(constituency_table['243774']))

    if not load_from_file:
        db.close()
        print('DB connection closed.')


def main():
    # JOB: Run settings:
    download_data = False
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
    data['datadate'] = pd.to_datetime(data['datadate'])

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
    # main()
    get_constituency_table(load_from_file=True)

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
