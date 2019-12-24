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
import math
import pandas as pd
import datetime
import numpy as np

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


def get_index_constituents(constituency_matrix: pd.DataFrame, date: datetime.date, folder_path: str) -> pd.Index:
    """
    Return company name list of index constituents for given date

    :param folder_path: Path to file directory
    :param constituency_matrix: Constituency table providing constituency information
    :param date: Date for which to return constituency list

    :return: Index of company identifiers for given date
    """

    lookup_dict = pd.read_json(os.path.join(folder_path, 'gvkey_name_dict.json'), typ='series').to_dict().get('conm')
    # print(lookup_dict)
    # print(len(constituency_matrix.loc[date].loc[lambda x: x == 1]))
    # print(constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values('gvkey').tolist())
    constituent_list_names = [lookup_dict.get(key) for key in
                              constituency_matrix.loc[date].loc[lambda x: x == 1].index.get_level_values(
                                  'gvkey').tolist()]

    return constituency_matrix.loc[date].loc[lambda x: x == 1].index


def retrieve_index_history(index_id: str = None, from_file=False, last_n: int = None,
                           folder_path: str = '', generate_dict=False) -> pd.DataFrame:
    """
    Download complete daily index history

    :return: DataFrame containing full index constituent data over full index history
    :rtype: pd.DataFrame
    """

    if not from_file:
        # Load GVKEYX lookup dict
        with open(os.path.join('data', 'gvkeyx_name_dict.json'), 'r') as fp:
            gvkeyx_lookup = json.load(fp)

        # Establish database connection
        print('Opening DB connection ...')
        db = wrds.Connection(wrds_username='afecker')
        print('Done')

        # Retrieve list of all stocks (gvkeys) for specified index including full date range of historic index data
        gvkey_list, relevant_date_range = get_all_constituents(
            constituency_matrix=pd.read_csv(os.path.join(folder_path, 'constituency_matrix.csv'), index_col=0,
                                            header=[0, 1],
                                            parse_dates=True))

        # Set start and end date
        if last_n:
            start_date = str(relevant_date_range[-last_n].date())
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
        print('Number of observations: %s' % data.shape[0])
        print('Number of individual dates: %d' % data.index.get_level_values('datadate').drop_duplicates().size)

        # JOB: Calculate Return Index Column
        data.loc[:, 'return_index'] = (data['prccd'] / data['ajexdi']) * data['trfd']

        # JOB: Calculate Daily Return
        data.loc[:, 'daily_return'] = data.groupby(level=['gvkey', 'iid'])['return_index'].apply(
            lambda x: x.pct_change(periods=1))

        # Reset index to date
        data = data.reset_index()

        # Save to file
        data.to_csv(os.path.join(folder_path, 'index_data_constituents.csv'))

    else:
        data = pd.read_csv(os.path.join(folder_path, 'index_data_constituents.csv'), dtype={'gvkey': str})

    if generate_dict:
        generate_company_lookup_dict(folder_path=folder_path, data=data)

    return data


def create_constituency_matrix(load_from_file=False, index_id='150095', folder_path: str = None) -> None:
    """
    Generate constituency matrix for stock market index components

    :param folder_path: Path to data folder
    :param index_id: Index to create constituency matrix for
    :param load_from_file: Flag indicating whether to load constituency information from file

    :return: Tuple containing [0] list of historical constituent gvkeys and [1] relevant date range
    """

    # File name with constituents and corresponding time frames
    file_name = 'data_constituents.csv'
    if folder_path:
        folder = folder_path
    else:
        folder = ''

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
        const_data.to_csv(os.path.join(folder, 'data_constituents.csv'))

    # JOB: Load table from local file
    else:
        # Load constituency table from file and transform key to string
        const_data = pd.read_csv(os.path.join(folder, file_name), dtype={'gvkey': str})
        # const_data['gvkey'] = const_data['gvkey'].astype('str')

        # Set gvkey and iid as MultiIndex
        const_data.set_index(['gvkey', 'iid'], inplace=True)

    # Convert date columns to datetime format
    for col in ['from', 'thru']:
        const_data.loc[:, col] = pd.to_datetime(const_data[col], format='%Y-%m-%d')
        const_data.loc[:, col] = const_data[col].dt.date

    # Determine period starting date and relevant date range
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
    constituency_matrix.to_csv(os.path.join(folder, 'constituency_matrix.csv'))

    if not load_from_file:
        db.close()
        print('DB connection closed.')

    return None


def get_all_constituents(constituency_matrix: pd.DataFrame) -> tuple:
    """
    Return all historical constituents' gvkeys and full date range from constituency matrix

    :param constituency_matrix: Matrix to extract constituents and time frame from
    :return: Tuple containing [0] list of historical constituent gvkeys and [1] relevant date range
    """

    # Query complete list of historic index constituents' gvkeys
    constituent_list = constituency_matrix.columns.get_level_values('gvkey').drop_duplicates().to_list()

    # Query relevant date range
    index_starting_date = constituency_matrix.index.sort_values()[0]
    relevant_date_range = pd.date_range(index_starting_date, datetime.date.today(), freq='D')

    return constituent_list, relevant_date_range


def generate_study_period(constituency_matrix: pd.DataFrame, full_data: pd.DataFrame,
                          period_range: tuple, index_name: str, folder_path: str) -> pd.DataFrame:
    """
    Generate a time-period sample for a study period

    :param folder_path: Path to data folder
    :param period_range: Date range of study period
    :type period_range: tuple
    :param full_data: Full stock data
    :type full_data: pd.DataFrame
    :param constituency_matrix: Constituency matrix
    :type constituency_matrix: pd.DataFrame
    :param index_name: Name of index
    :type index_name: str

    :return: Study period sample
    :rtype: pd.DataFrame
    """

    # Convert date columns to DatetimeIndex
    full_data['datadate'] = pd.to_datetime(full_data['datadate'])

    # Get list of constituents for specified date
    full_data.set_index('datadate', inplace=True)
    unique_dates = full_data.index.drop_duplicates()
    constituent_indices = get_index_constituents(constituency_matrix, unique_dates[period_range[1]],
                                                 folder_path=folder_path)

    full_data.reset_index(inplace=True)

    print('Retrieving index constituency for %s as of %s' % (index_name, unique_dates[period_range[1]]))

    # Select relevant data
    full_data = full_data.set_index(['gvkey', 'iid'])
    full_data = full_data.loc[constituent_indices, :]
    full_data = full_data.reset_index()
    full_data.set_index('datadate', inplace=True)
    full_data.sort_index(inplace=True)

    # Select data from study period
    print(unique_dates[period_range[0]])
    print(unique_dates[period_range[1]])
    print(
        'Retrieving data from %s to %s' % (unique_dates[period_range[0]].date(), unique_dates[period_range[1]].date()))
    study_data = full_data.loc[unique_dates[period_range[0]]:unique_dates[period_range[1]]]

    # JOB: Add standardized daily returns
    mean_daily_return = study_data.loc[unique_dates[period_range[0]]:unique_dates[period_range[1]],
                        'daily_return'].mean()
    std_daily_return = study_data.loc[unique_dates[period_range[0]]:unique_dates[period_range[1]], 'daily_return'].std()
    print('Mean daily return: %g' % mean_daily_return)
    print('Std. daily return: %g' % std_daily_return)

    # JOB: Fill n/a values for trading volume
    study_data.loc[:, 'cshtrd'] = study_data.loc[:, 'cshtrd'].fillna(value=0)

    # JOB: Calculate standardized daily returns
    study_data.loc[:, 'stand_d_return'] = (study_data.loc[:, 'daily_return'] - mean_daily_return) / std_daily_return

    # JOB: Create target
    study_data.loc[:, 'above_cs_med'] = study_data.loc[:, 'daily_return'].gt(
        study_data.groupby('datadate')['daily_return'].transform('median')).astype(int)
    study_data.loc[:, 'cs_med'] = study_data.groupby('datadate')['daily_return'].transform('median')

    # JOB: Create cross-sectional ranking
    study_data.loc[:, 'cs_rank'] = study_data.groupby('datadate')['daily_return'].rank(method='first').astype('int8')
    study_data.loc[:, 'cs_percentile'] = study_data.groupby('datadate')['daily_return'].rank(pct=True)

    # JOB: Number of securities in cross-section
    study_data.loc[:, 'cs_length'] = study_data.groupby('datadate')['daily_return'].count()

    return study_data


def generate_index_lookup_dict() -> None:
    """
    Generate dictionary mapping GVKEYX (index identifier) to index name

    :return:
    """

    gvkeyx_lookup = pd.read_csv(os.path.join('data', 'Compustat_Global_Indexes.csv'), index_col='GVKEYX')
    gvkeyx_lookup.index = gvkeyx_lookup.index.astype(str)
    gvkeyx_lookup = gvkeyx_lookup['index_name'].to_dict()

    with open(os.path.join('data', 'gvkeyx_name_dict.json'), 'w') as fp:
        json.dump(gvkeyx_lookup, fp)


def generate_company_lookup_dict(folder_path: str, data: pd.DataFrame) -> None:
    """
    Generate dictionary mapping GVKEY (stock identifier) to stock name

    :return:
    """

    # JOB: Create company name dictionary and save to file
    company_name_lookup = data.reset_index()
    company_name_lookup = company_name_lookup.loc[
        ~company_name_lookup.duplicated(subset='gvkey', keep='first'), ['gvkey', 'conm']].set_index('gvkey')

    # JOB: Save dict to json file
    company_name_lookup.to_json(os.path.join(folder_path, 'gvkey_name_dict.json'))


def main():
    # JOB: Run settings:
    download_data = True
    pivot_transform = False
    data = None

    # Establish database connection
    print('Opening DB connection ...')
    db = wrds.Connection(wrds_username='afecker')
    print('Done')

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

    # Close database connection
    db.close()
    print('DB connection closed.')


# Main method
if __name__ == '__main__':
    generate_index_lookup_dict()
    pass
    # main()
    # create_constituency_matrix(load_from_file=False)
    # retrieve_index_history(index_id='150095', from_file=False, last_n=None)
