__author__ = "Sven Köpke"
__copyright__ = "Sven Köpke 2019"
__version__ = "0.0.1"
__license__ = "MIT"

import datetime
import json
import random
from typing import Union, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import binary_accuracy

import config
from config import *
from core.data_collection import generate_study_period
from core.data_processor import DataLoader
from core.model import LSTMModel, TreeEnsemble, WeightedEnsemble, MixedEnsemble
from core.utils import plot_train_val, get_most_recent_file, CSVWriter, \
    check_data_conformity, annualize_metric, Timer, calc_sharpe, \
    calc_excess_returns, calc_sortino, add_to_json, get_model_parent_type, ProgressBar, check_directory_for_file, \
    pretty_print_table, df_to_html


def main(index_id='150095', index_name='', full_data=None, constituency_matrix: pd.DataFrame = None,
         folder_path: str = None,
         columns: list = None, data_only=False,
         load_last: bool = False,
         start_index: int = -1001, end_index: int = -1, model_type: str = None,
         ensemble: Union[WeightedEnsemble, MixedEnsemble] = None,
         verbose=2, plotting=False):
    """
    Run data preparation and model training

    :param plotting:
    :param ensemble:
    :param folder_path:
    :param constituency_matrix:
    :param full_data:
    :param index_name:
    :param verbose: Verbosity
    :param model_type: Model type as strings: 'deep_learning' for LSTM, 'tree_based' for Random Forest
    :param columns: Relevant columns for model training and testing
    :param load_last: Flag indicating whether to load last model weights from storage
    :param index_id: Index identifier
    :param data_only: Flag indicating whether to download data only without training the model
    :param start_index: Index of period start date
    :param end_index: Index of period end date
    :return: None
    """

    # JOB: Load configurations
    configs = json.load(open(os.path.join(ROOT_DIR, 'config.json'), 'r'))

    # Retrieve parent mode type from dict
    parent_model_type = get_model_parent_type(configs=configs, model_type=model_type)

    if isinstance(ensemble, WeightedEnsemble):
        parent_model_type = 'tree_based'
    elif isinstance(ensemble, MixedEnsemble):
        parent_model_type = 'mixed'

    # Add 'return_index' and remove 'stand_d_return' column for feature generation in case of tree-based model
    if parent_model_type == 'tree_based':
        columns.extend(['return_index'])
        columns.remove('stand_d_return')

    if parent_model_type == 'mixed':
        return_validation_data = True
    else:
        return_validation_data = False

    # Delete columns containing 'unnamed' and save file
    if any('unnamed' in s.lower() for s in full_data.columns):
        full_data = full_data.get([column for column in full_data.columns if not column.startswith('Unnamed')])
        full_data.to_csv(os.path.join(ROOT_DIR, folder_path, 'index_data_constituents.csv'))

    full_data.dropna(subset=['daily_return'], inplace=True)  # Drop records without daily return data

    # JOB: Query number of dates in full data set
    data_length = full_data['datadate'].drop_duplicates().shape[0]  # Number of individual dates

    if data_only:
        print(f'Finished downloading data for {index_name}.')
        print(f'Data set contains {data_length} individual dates.')
        # print(full_data.head(10))
        # print(full_data.tail(10))
        return full_data

    # JOB: Specify study period interval
    period_range = (start_index, end_index)

    # JOB: Get study period data and split index
    try:
        study_period_data, split_index = generate_study_period(constituency_matrix=constituency_matrix,
                                                               full_data=full_data.copy(),
                                                               period_range=period_range,
                                                               index_name=index_name, configs=configs,
                                                               folder_path=folder_path, columns=columns)
    except AssertionError as ae:
        print(ae)
        print('Not all constituents in full data set. Terminating execution.')
        return
    except RuntimeWarning as rtw:
        print(rtw)
        return

    # Delete full data copy
    del full_data

    # JOB: Get all dates in study period
    full_date_range = study_period_data.index.unique()
    print(f'Study period length: {len(full_date_range)}\n')
    start_date = full_date_range.min().date()
    end_date = full_date_range.max().date()

    print(study_period_data.index.name)
    # JOB: Set MultiIndex to stock identifier
    study_period_data = study_period_data.reset_index().set_index(['gvkey', 'iid'])

    # Get unique stock indices in study period
    unique_indices = study_period_data.index.unique()

    # pretty_print_table(study_period_data.iloc[:20, :])
    # df_to_html(study_period_data.tail(20)[
    #                ['datadate', 'ajexdi', 'trfd', 'cshtrd', 'conml', 'gics_sector_name', 'prcld', 'prchd', 'prccd',
    #                 'return_index',
    #                 'daily_return', 'stand_d_return', 'cs_med', 'above_cs_med', 'cs_length']], title='Raw Data',
    #            open_window=True)


    # Calculate Sharpe Ratio for full index history
    sharpe_ratio_sp = calc_sharpe(study_period_data.set_index('datadate').loc[:, ['daily_return']], annualize=True)

    print(f'Annualized Sharpe Ratio: {np.round(sharpe_ratio_sp, 4)}')
    validation_range = full_date_range[random.sample(range(split_index + 1), int(0.25 * split_index))]

    # JOB: Obtain training and test data as well as test data index and feature names
    x_train, y_train, x_test, y_test, train_data_index, test_data_index, feature_names, validation_data = preprocess_data(
        study_period_data=study_period_data.copy(),
        split_index=split_index,
        unique_indices=unique_indices, cols=columns,
        configs=configs,
        full_date_range=full_date_range,
        parent_model_type=parent_model_type,
        ensemble=ensemble, return_validation_data=return_validation_data, validation_range=validation_range)

    if parent_model_type != 'mixed':
        print(f'\nLength of training data: {x_train.shape}')
        print(f'Length of test data: {x_test.shape}')
        print(f'Length of test target data: {y_test.shape}')

        target_mean_train, target_mean_test = check_data_conformity(x_train=x_train, y_train=y_train, x_test=x_test,
                                                                    y_test=y_test, test_data_index=test_data_index)

        print(f'Average target label (training): {np.round(target_mean_train, 4)}')
        print(f'Average target label (test): {np.round(target_mean_test, 4)}\n')
        print(f'Performance validation thresholds: \n'
              f'Training: {np.round(1 - target_mean_train, 4)}\n'
              f'Testing: {np.round(1 - target_mean_test, 4)}\n')

    history = None
    predictions = None
    model = None
    test_data_index_merged = None

    # JOB: Test model performance on test data
    if parent_model_type == 'deep_learning':
        # JOB: Load model from storage
        if load_last:
            model: LSTMModel = LSTMModel()
            model.load_model(get_most_recent_file('saved_models'))

        # JOB: Build model from configs
        else:
            model: LSTMModel = LSTMModel(index_name=index_name.lower().replace(' ', '_'))
            model.build_model(configs, verbose=2)

            # JOB: In-memory training
            history = model.train(
                x_train,
                y_train,
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir'], configs=configs, verbose=verbose
            )

        best_val_accuracy = float(history.history['val_accuracy'][np.argmin(history.history['val_loss'])])

        print(f'Validation accuracy of best model: {np.round(best_val_accuracy, 3)}')

        # JOB: Make point prediction
        predictions = model.predict(x_test)

    elif parent_model_type == 'tree_based':
        if model_type:  # Single tree-based classifier
            model: TreeEnsemble = TreeEnsemble(index_name=index_name.lower().replace(' ', '_'), model_type=model_type)
            model.build_model(configs=configs, verbose=verbose)

            # JOB: Fit model
            print(f'\n\nFitting {model_type} model ...')
            timer = Timer().start()
            model.fit(x_train, y_train)

            try:
                print('\nFeature importances:')
                print(model.model.feature_importances_)
            except AttributeError as ae:
                pass

            timer.stop()

            print(f'\nMaking predictions on test set ...')
            timer = Timer().start()
            predictions = model.model.predict_proba(x_test)[:, 1]
            timer.stop()

        elif ensemble:  # Ensemble of tree-based classifiers
            model: WeightedEnsemble = ensemble
            print('Ensemble configuration:')
            print(model)
            model_type = f'Ensemble({", ".join(ensemble.classifier_types)})'

            # JOB: Fit models
            print(f'\n\nFitting ensemble ...')
            model.fit(x_train, y_train, feature_names=feature_names, show_importances=False)

            print(f'\nOut-of-bag scores: {model.oob_scores}')

            weighting_whitelist = ['RandomForestClassifier', 'ExtraTreesClassifier']
            if all([clf_type in weighting_whitelist for clf_type in ensemble.classifier_types]):
                weighted = True
            else:
                weighted = False

            predictions = model.predict(x_test, weighted=weighted, oob_scores=model.oob_scores)

            # JOB: Testing ensemble components and ensemble performance
            print('Testing ensemble components:')

            all_predictions = model.predict_all(x_test)
            for i, base_clf in enumerate(ensemble.classifiers):
                print(f'Testing ensemble component {i + 1} ({base_clf.model_type})')
                test_model(predictions=all_predictions[i],
                           configs=configs, folder_path=folder_path, test_data_index=test_data_index,
                           y_test=y_test, study_period_data=study_period_data.copy(),
                           parent_model_type=parent_model_type,
                           model_type=base_clf.model_type, history=history,
                           index_id=index_id, index_name=index_name, study_period_length=len(full_date_range),
                           model=model.classifiers[i],
                           period_range=period_range, start_date=start_date, end_date=end_date, plotting=plotting)

    elif parent_model_type == 'mixed':
        # JOB: Fitting and testing in case of mixed ensemble
        model: MixedEnsemble = ensemble
        print('Mixed ensemble configuration:')
        print(model)

        x_val = []
        y_val = []
        val_index = []

        for i in range(2):
            x_val_i, y_val_i, val_index_i = validation_data[i]
            x_val.append(x_val_i)
            y_val.append(y_val_i)
            val_index.append(val_index_i)

        # JOB: Fit models
        print(f'\n\nFitting ensemble  ...')
        model.fit(x_train, y_train, feature_names=feature_names, x_val=x_val, val_index=val_index,
                  configs=configs,
                  folder_path=folder_path, y_test=y_val, test_data_index=val_index,
                  study_period_data=study_period_data.copy(), weighting_criterion='Mean Daily Excess Return')

        print(f'\nValidation/Out-of-bag scores: {model.val_scores}')

        # JOB: Testing ensemble components and ensemble performance
        print('Testing ensemble components:')

        all_predictions, y_test_merged, test_data_index_merged = model.predict_all(x_test, y_test=y_test,
                                                                                   test_data_index=test_data_index)
        for i, base_clf in enumerate(ensemble.classifiers):
            print(f'Testing ensemble component {i + 1} ({base_clf.model_type})')
            test_model(predictions=all_predictions[i],
                       configs=configs, folder_path=folder_path, test_data_index=test_data_index_merged,
                       y_test=y_test_merged, study_period_data=study_period_data.copy(),
                       parent_model_type=base_clf.parent_model_type,
                       model_type=base_clf.model_type, history=history,
                       index_id=index_id, index_name=index_name, study_period_length=len(full_date_range),
                       model=model.classifiers[i],
                       period_range=period_range, start_date=start_date, end_date=end_date, plotting=plotting)

        print('Done testing individual models.')

        predictions, y_test, test_data_index_merged = model.predict(x_test=x_test, y_test_merged=y_test_merged,
                                                                    all_predictions=all_predictions, weighted=True,
                                                                    test_data_index=test_data_index,
                                                                    test_data_index_merged=test_data_index_merged,
                                                                    y_test=y_test)

    # preds = pd.DataFrame({'performance': [], 'rank': []})

    # JOB: Test model
    if isinstance(predictions[0], tuple):
        for weighting, sub_predictions in predictions:
            pred = test_model(predictions=sub_predictions, configs=configs, folder_path=folder_path,
                              test_data_index=test_data_index_merged,
                              y_test=y_test, study_period_data=study_period_data.copy(),
                              parent_model_type=parent_model_type,
                              model_type=f'{model.model_type}_{weighting}', history=history,
                              index_id=index_id, index_name=index_name, study_period_length=len(full_date_range),
                              model=model,
                              period_range=period_range, start_date=start_date, end_date=end_date, plotting=plotting)

            # preds.loc[:, weighting] = pred
    else:
        test_model(predictions=predictions, configs=configs, folder_path=folder_path, test_data_index=test_data_index,
                   y_test=y_test, study_period_data=study_period_data.copy(), parent_model_type=parent_model_type,
                   model_type=model.model_type, history=history,
                   index_id=index_id, index_name=index_name, study_period_length=len(full_date_range), model=model,
                   period_range=period_range, start_date=start_date, end_date=end_date, plotting=plotting)

    # preds.loc[:, 'Same'] = np.where(preds['performance'] == preds['rank'], True, False)
    # print(f"{preds['Same'].sum()}/{preds.shape[0]}")  # TODO: Remove

    del study_period_data


def preprocess_data(study_period_data: pd.DataFrame, unique_indices: pd.MultiIndex, cols: list, split_index: int,
                    configs: dict, full_date_range: pd.Index, parent_model_type: str,
                    ensemble: Union[WeightedEnsemble, MixedEnsemble] = None, from_mixed=False,
                    return_validation_data=False, validation_range: list = None) -> tuple:
    """
    Pre-process study period data to obtain training and test sets as well as test data index

    :param validation_range:
    :param return_validation_data: Whether to return tuple of validation data
    :param from_mixed: Flag indicating that data is processed for MixedEnsemble component
    :param ensemble: Ensemble instance
    :param parent_model_type: 'deep_learning' or 'tree_based'
    :param study_period_data: Full data for study period
    :param unique_indices: MultiIndex of unique indices in study period
    :param cols: Relevant columns
    :param split_index: Split index
    :param configs: Configuration dict
    :param full_date_range: Index of dates
    :return: Tuple of (x_train, y_train, x_test, y_test, test_data_index, data_cols, [Optional](validation data tuple))
    """

    cols = cols.copy()
    print(f'{Style.BRIGHT}{Fore.YELLOW}Pre-processing data ...{Style.RESET_ALL}')
    timer = Timer().start()

    if from_mixed and parent_model_type == 'tree_based':
        cols.extend(['return_index'])
        cols.remove('stand_d_return')

    if parent_model_type == 'mixed':
        # JOB: Pre-process data separately for each mixed ensemble component
        ensemble_components_parent_types = [get_model_parent_type(configs, clf) for clf in ensemble.classifier_types]
        return np.array([preprocess_data(
            study_period_data=study_period_data, unique_indices=unique_indices, cols=cols, split_index=split_index,
            configs=configs, full_date_range=full_date_range, parent_model_type=parent_model_type,
            ensemble=None, from_mixed=True, return_validation_data=True, validation_range=validation_range) for
            parent_model_type in ['deep_learning', 'tree_based']]).T.tolist()

    # JOB: Instantiate training and test data
    data = None
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None
    validation_mask = []
    train_data_index = pd.Index([])
    validation_data_index = pd.Index([])
    test_data_index = pd.Index([])

    train_range = full_date_range[configs['data']['sequence_length'] + 1:split_index + 1]
    print(f'Last training date: {full_date_range[split_index]}')

    print(f'Training range length: {len(train_range)}')

    if return_validation_data:
        print(f'Validation range length: {len(validation_range)}')

    # Instantiate progress bar
    progress_bar = ProgressBar(len(unique_indices))
    # JOB: Iterate through individual stocks and generate training and test data
    for stock_id in unique_indices:
        progress_bar.print_progress()
        id_data = study_period_data.loc[stock_id].set_index('datadate')

        try:
            # JOB: Initialize DataLoader
            data = DataLoader(
                id_data, cols=cols,
                seq_len=configs['data']['sequence_length'], full_date_range=full_date_range, stock_id=stock_id,
                split_index=split_index, model_type=parent_model_type
            )
        except AssertionError as ae:
            print(ae)
            continue

        # print('Length of data for ID %s: %d' % (stock_id, len(id_data)))

        # JOB: Generate training data
        x_train_i, y_train_i = data.get_train_data()

        # JOB: Generate test data
        x_test_i, y_test_i = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
        )

        # JOB: In case training/test set is empty, set to first batch, otherwise append data
        if (len(x_train_i) > 0) and (len(y_train_i) > 0):
            train_data_index = train_data_index.append(data.data_train_index)
            mask = [date_i in validation_range for date_i in data.data_train_index.get_level_values(level=0)]
            validation_mask.extend(mask)
            if x_train is None:
                x_train = x_train_i
                y_train = y_train_i
            else:
                x_train = np.append(x_train, x_train_i, axis=0)
                y_train = np.append(y_train, y_train_i, axis=0)

        if (len(x_test_i) > 0) and (len(y_test_i) > 0):
            test_data_index = test_data_index.append(data.data_test_index)
            if x_test is None:
                x_test = x_test_i
                y_test = y_test_i
            else:
                x_test = np.append(x_test, x_test_i, axis=0)
                y_test = np.append(y_test, y_test_i, axis=0)

            if len(y_test_i) != len(data.data_test_index):
                print(f'\nMismatch for index {stock_id}')
                print(f'{Fore.YELLOW}{Back.RED}{Style.BRIGHT}Lengths do not conform!{Style.RESET_ALL}')
                print(f'Target length: {len(y_test_i)}, index length: {len(data.data_test_index)}')
                raise AssertionError('Data length is not conforming.')

        # print('Length of test labels: %d' % len(y_test))
        # print('Length of test index: %d\n' % len(test_data_index))

        if len(y_test) != len(test_data_index):
            raise AssertionError('Data length is not conforming.')

    print(f'Length training data (before): {len(x_train)}')

    if return_validation_data:
        # JOB: Split up training data in training and validation set if needed
        inverse_mask = [not m for m in validation_mask]
        x_val = x_train[validation_mask]
        y_val = y_train[validation_mask]
        validation_data_index = train_data_index[validation_mask]
        x_train = x_train[inverse_mask]
        y_train = y_train[inverse_mask]
        train_data_index = train_data_index[inverse_mask]

        print(f'Length x_train: {len(x_train)}')
        print(f'Length x_val: {len(x_val)}')

    print(f'Length x_train: {len(x_train)}')

    print(f'{Style.BRIGHT}{Fore.YELLOW}Done pre-processing data.{Style.RESET_ALL}')
    timer.stop()

    return x_train, y_train, x_test, y_test, train_data_index, test_data_index, data.cols, (
        x_val, y_val, validation_data_index)


def test_model(predictions: np.array, configs: dict, folder_path: str, test_data_index: pd.Index,
               y_test: np.array,
               study_period_data: pd.DataFrame, parent_model_type: str = 'deep_learning', model_type: str = None,
               history=None, index_id='',
               index_name='', study_period_length: int = 0, model=None, period_range: tuple = (0, 0),
               start_date: datetime.date = datetime.date.today(), end_date: datetime.date = datetime.date.today(),
               get_val_score_only=False, weighting_criterion=None, plotting=False, **kwargs):
    """
    Test model on unseen data.

    :param plotting:
    :param weighting_criterion:
    :param get_val_score_only:
    :param model_type:
    :param predictions:
    :param configs:
    :param folder_path:
    :param test_data_index:
    :param y_test:
    :param study_period_data:
    :param parent_model_type:
    :param history:
    :param index_id:
    :param index_name:
    :param study_period_length:
    :param model:
    :param period_range:
    :param start_date:
    :param end_date:
    """

    if get_val_score_only:
        # In case classifier is part of MixedEnsemble as is being validated
        y_test = y_test[kwargs['model_index']]
        test_data_index = test_data_index[kwargs['model_index']]
        print(f'\nGetting validation score for {Style.BRIGHT}{Fore.BLUE}{model_type}{Style.RESET_ALL} ...')
    else:
        print(f'\nTesting {Style.BRIGHT}{Fore.BLUE}{model_type}{Style.RESET_ALL} model on unseen data ...')

    # print(f'{Style.BRIGHT}{Fore.MAGENTA}Length of test data: {len(y_test)}{Style.RESET_ALL}')

    study_period_data = study_period_data.copy()

    timer = Timer().start()
    # JOB: Create data frame with true and predicted values
    if isinstance(test_data_index, pd.MultiIndex):
        test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                           index=test_data_index)

    else:
        test_set_comparison = pd.DataFrame({'y_test': y_test.astype('int8').flatten(), 'prediction': predictions},
                                           index=pd.MultiIndex.from_tuples(test_data_index,
                                                                           names=['datadate', 'stock_id']))

    # JOB: Transform index of study period data to match test_set_comparison index
    study_period_data.index = study_period_data.index.tolist()  # Flatten MultiIndex to tuples
    study_period_data.index.name = 'stock_id'  # Rename index
    study_period_data.set_index('datadate', append=True, inplace=True)

    # JOB: Merge test set with study period data
    test_set_comparison = test_set_comparison.merge(study_period_data, how='left', left_index=True,
                                                    right_on=['datadate', 'stock_id'])

    del study_period_data

    full_portfolio = None

    # JOB: Create normalized predictions (e.g., directional prediction relative to cross-sectional median of predictions)
    test_set_comparison.loc[:, 'norm_prediction'] = test_set_comparison.loc[:, 'prediction'].gt(
        test_set_comparison.groupby('datadate')['prediction'].transform('median')).astype(np.int16)

    # JOB: Create cross-sectional ranking
    test_set_comparison.loc[:, 'prediction_rank'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        method='first', ascending=False).astype('int16')
    test_set_comparison.loc[:, 'prediction_percentile'] = test_set_comparison.groupby('datadate')['prediction'].rank(
        pct=True)

    test_data_start_date = test_set_comparison.index.get_level_values('datadate').min().date()
    test_data_end_date = test_set_comparison.index.get_level_values('datadate').max().date()
    test_set_n_days = test_set_comparison.index.get_level_values('datadate').unique().size
    test_set_n_constituents = test_set_comparison.index.get_level_values('stock_id').unique().size

    cross_section_size = int(round(test_set_comparison.groupby('datadate')['y_test'].count().mean()))
    print(f'Average size of cross sections: {int(cross_section_size)}')

    # Define top k values
    top_k_list = [5, 10]

    if cross_section_size > 30:
        top_k_list.extend([50, 100, 150, 200, 250])

    # JOB: Create empty dataframe for recording top-k accuracies
    top_k_metrics = pd.DataFrame(
        {'Accuracy': [], 'Mean Daily Return': [], 'Mean Daily Return (Short)': [], 'Mean Daily Return (Long)': []})
    top_k_metrics.index.name = 'k'

    t_costs = 0.0005

    market_metrics = None
    if not get_val_score_only:
        market_metrics, market_cum_returns = get_market_metrics(test_set_comparison, t_costs=t_costs)

    top_10_return_series = None
    top_10_error_series = None

    for top_k in top_k_list:
        # JOB: Filter test data by top/bottom k affiliation
        long_positions = test_set_comparison[test_set_comparison['prediction_rank'] <= top_k]
        short_positions = test_set_comparison[test_set_comparison['prediction_rank'] > cross_section_size - top_k]
        short_positions.loc[:, 'daily_return'] = - short_positions.loc[:, 'daily_return']

        full_portfolio = pd.concat([long_positions, short_positions], axis=0)

        # print(full_portfolio.head())
        # print(full_portfolio.tail())

        if not get_val_score_only:
            market_metrics, market_cum_returns = get_market_metrics(test_set_comparison, t_costs=t_costs)

            return_series = full_portfolio.loc[:, 'daily_return'].groupby(level=['datadate']).mean()
            if top_k == 10:
                top_10_return_series = return_series
                top_10_return_series = top_10_return_series.reset_index()
                top_10_return_series.loc[:, 'datadate'] = top_10_return_series['datadate'].dt.strftime('%Y-%m-%d')
                top_10_return_series.set_index('datadate', inplace=True)

                sorted_portfolio = full_portfolio.set_index('prediction_rank', append=True, inplace=False)
                sorted_portfolio.reset_index(['stock_id'], inplace=True)
                sorted_portfolio.sort_index(level=['datadate', 'prediction_rank'], inplace=True)
                sorted_portfolio.reset_index(level='datadate', inplace=True, drop=True)
                top_10_error_series = (sorted_portfolio['norm_prediction'] - sorted_portfolio['y_test']).abs()
                top_10_error_series = top_10_error_series.values.tolist()

            cumulative_return = (return_series + 1).cumprod().rename('Cumulative Portfolio Return')
            cumulative_return.index.name = 'Time'

            merged = pd.concat([cumulative_return, market_cum_returns], axis=1, join='outer')
            if plotting:
                merged.plot()
                plt.legend(loc='best')
                plt.title(label=model_type)
                plt.show()

        annualized_sharpe = calc_sharpe(full_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean(),
                                        annualize=True)
        annualized_sharpe_atc = calc_sharpe(
            full_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean() - 4 * t_costs,
            annualize=True)
        annualized_sortino = calc_sortino(full_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean(),
                                          annualize=True)
        annualized_sortino_atc = calc_sortino(
            full_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean() - 4 * t_costs,
            annualize=True)

        accuracy = None

        # JOB: Calculate accuracy score over all trades
        if parent_model_type == 'deep_learning':
            accuracy = binary_accuracy(full_portfolio['y_test'].values,
                                       full_portfolio['norm_prediction'].values).numpy()

        elif parent_model_type == 'tree_based':
            accuracy = accuracy_score(full_portfolio['y_test'].values,
                                      full_portfolio['norm_prediction'].values)

        elif parent_model_type == 'mixed':
            accuracy = accuracy_score(full_portfolio['y_test'].values,
                                      full_portfolio['norm_prediction'].values)

        mean_daily_return = full_portfolio.groupby(level=['datadate'])['daily_return'].mean().mean()

        mean_daily_excess_return = calc_excess_returns(
            full_portfolio.groupby(level=['datadate'])['daily_return'].mean().rename('daily_return')).mean()

        mean_daily_short = short_positions.groupby(level=['datadate'])['daily_return'].mean().mean()
        mean_daily_long = long_positions.groupby(level=['datadate'])['daily_return'].mean().mean()

        top_k_metrics.loc[top_k, 'Accuracy'] = accuracy
        top_k_metrics.loc[top_k, 'Mean Daily Return'] = mean_daily_return
        top_k_metrics.loc[top_k, 'Annualized Return'] = annualize_metric(mean_daily_return)
        top_k_metrics.loc[top_k, 'Mean Daily Excess Return'] = mean_daily_excess_return
        top_k_metrics.loc[top_k, 'Annualized Excess Return'] = annualize_metric(mean_daily_excess_return)
        top_k_metrics.loc[top_k, 'Annualized Sharpe'] = annualized_sharpe
        top_k_metrics.loc[top_k, 'Annualized Sortino'] = annualized_sortino
        top_k_metrics.loc[top_k, 'Mean Daily Return (Short)'] = mean_daily_short
        top_k_metrics.loc[top_k, 'Mean Daily Return (Long)'] = mean_daily_long

        # JOB: Add metrics incl. transaction costs of 5 bps per half-turn
        top_k_metrics.loc[top_k, 'Mean Daily Return_atc'] = mean_daily_return - 4 * t_costs
        top_k_metrics.loc[top_k, 'Annualized Return_atc'] = annualize_metric(mean_daily_return - 4 * t_costs)
        top_k_metrics.loc[top_k, 'Mean Daily Excess Return_atc'] = mean_daily_excess_return - 4 * t_costs
        top_k_metrics.loc[top_k, 'Annualized Excess Return_atc'] = annualize_metric(
            mean_daily_excess_return - 4 * t_costs)
        top_k_metrics.loc[top_k, 'Annualized Sharpe_atc'] = annualized_sharpe_atc
        top_k_metrics.loc[top_k, 'Annualized Sortino_atc'] = annualized_sortino_atc
        top_k_metrics.loc[top_k, 'Mean Daily Return (Short)_atc'] = mean_daily_short - 2 * t_costs
        top_k_metrics.loc[top_k, 'Mean Daily Return (Long)_atc'] = mean_daily_long - 2 * t_costs

    if get_val_score_only:
        print(f'{weighting_criterion} score: {top_k_metrics.loc[10, weighting_criterion]}')
        return top_k_metrics.loc[10, weighting_criterion]

    top_k_metrics = pd.concat([top_k_metrics, market_metrics.to_frame().T], join='outer', verify_integrity=True)
    top_k_metrics.fillna('-', inplace=True)

    # JOB: Display top-k metrics
    pretty_print_table(top_k_metrics)

    # JOB: Plot accuracies and save figure to file
    if plotting:
        for col in top_k_metrics.columns:
            top_k_metrics[col].plot(kind='line', legend=True, fontsize=14)
            plt.savefig(os.path.join(ROOT_DIR, folder_path, f'top_k_{col.lower()}.png'), dpi=600)
            plt.show()

        if parent_model_type == 'deep_learning':
            # JOB: Plot training and validation metrics for LSTM
            try:
                plot_train_val(history, configs['model']['metrics'], store_png=True, folder_path=folder_path)
            except AttributeError as ae:
                print(f'{Fore.RED}{Style.BRIGHT}Plotting failed.{Style.RESET_ALL}')
                # print(ae)
            except UnboundLocalError as ule:
                print(
                    f'{Fore.RED}{Back.YELLOW}{Style.BRIGHT}Plotting failed. History has not been created.{Style.RESET_ALL}')
                # print(ule)

    # JOB: Evaluate model on full test data
    test_score = None
    if parent_model_type == 'deep_learning':
        test_score = float(binary_accuracy(test_set_comparison['y_test'].values,
                                           test_set_comparison['norm_prediction'].values).numpy())

        print(f'\nTest score on full test set: {float(np.round(test_score, 4))}')

    elif parent_model_type in ['tree_based', 'mixed']:
        test_score = accuracy_score(test_set_comparison['y_test'].values,
                                    test_set_comparison['norm_prediction'].values)
        print(f'\nTest score on full test set: {np.round(test_score, 4)}')

        # pretty_print_table(
        #     pd.DataFrame({'y_test': test_set_comparison['y_test'].values, 'norm_prediction': test_set_comparison[
        #         'norm_prediction'].values}).sample(100))  # TODO: Remove

    total_epochs = len(history.history['loss']) if history is not None else None

    # JOB: Fill dict for logging
    data_record = {
        'ID': config.run_id,
        'Experiment Run End': datetime.datetime.now().isoformat(),
        'Parent Model Type': parent_model_type,
        'Model Type': model_type,
        'Index ID': index_id,
        'Index Name': index_name,
        'Study Period Length': study_period_length,
        'Period Range': period_range,
        'Study Period Start Date': start_date.isoformat(),
        'Study Period End Date': end_date.isoformat(),
        'Test Set Size': y_test.shape[0],
        'Days Test Set': test_set_n_days,
        'Constituent Number': test_set_n_constituents,
        'Average Cross Section Size': cross_section_size,
        'Test Set Start Date': test_data_start_date.isoformat(),
        'Test Set End Date': test_data_end_date.isoformat(),
        'Total Accuracy': test_score,

        'Top-k Accuracy Scores': top_k_metrics['Accuracy'].to_dict(),
        'Top-k Mean Daily Return': top_k_metrics['Mean Daily Return'].to_dict(),
        'Top-k Mean Daily Excess Return': top_k_metrics['Mean Daily Excess Return'].to_dict(),
        'Top-k Annualized Excess Return': top_k_metrics['Annualized Excess Return'].to_dict(),
        'Top-k Annualized Return': top_k_metrics['Annualized Return'].to_dict(),
        'Top-k Annualized Sharpe': top_k_metrics['Annualized Sharpe'].to_dict(),
        'Top-k Annualized Sortino': top_k_metrics['Annualized Sortino'].to_dict(),
        'Mean Daily Return (Short)': top_k_metrics['Mean Daily Return (Short)'].to_dict(),
        'Mean Daily Return (Long)': top_k_metrics['Mean Daily Return (Long)'].to_dict(),

        'Top-k Mean Daily Return_atc': top_k_metrics['Mean Daily Return_atc'].to_dict(),
        'Top-k Annualized Return_atc': top_k_metrics['Annualized Return_atc'].to_dict(),
        'Top-k Mean Daily Excess Return_atc': top_k_metrics['Mean Daily Excess Return_atc'].to_dict(),
        'Top-k Annualized Excess Return_atc': top_k_metrics['Annualized Excess Return_atc'].to_dict(),
        'Top-k Annualized Sharpe_atc': top_k_metrics['Annualized Sharpe_atc'].to_dict(),
        'Top-k Annualized Sortino_atc': top_k_metrics['Annualized Sortino_atc'].to_dict(),
        'Top-k Mean Daily Return (Short)_atc': top_k_metrics['Mean Daily Return (Short)_atc'].to_dict(),
        'Top-k Mean Daily Return (Long)_atc': top_k_metrics['Mean Daily Return (Long)_atc'].to_dict(),

        'Model Configs': model.get_params(),
        'Total Epochs': total_epochs,

        'Return Series': top_10_return_series.to_dict(),
        'Prediction Error': top_10_error_series
    }

    data_record_json = data_record

    forest_record = {'Experiment Run End': datetime.datetime.now().isoformat(),
                     'Parent Model Type': str(parent_model_type),
                     'Model Type': str(model_type),
                     'Index ID': str(index_id),
                     'Index Name': index_name,
                     'Study Period Length': study_period_length,
                     'Test Set Size': y_test.shape[0],
                     'Days Test Set': test_set_n_days,
                     'Constituent Number': test_set_n_constituents,
                     'Average Cross Section Size': cross_section_size,
                     'Test Set Start Date': test_data_start_date.isoformat(),
                     'Test Set End Date': test_data_end_date.isoformat(),
                     'Total Accuracy': test_score,
                     'Top-k Accuracy Scores': top_k_metrics.loc[10, 'Accuracy'],
                     'Top-k Mean Daily Return': top_k_metrics.loc[10, 'Mean Daily Return'],
                     'Top-k Mean Daily Excess Return': top_k_metrics.loc[10, 'Mean Daily Excess Return'],
                     'Top-k Annualized Excess Return': top_k_metrics.loc[10, 'Annualized Excess Return'],
                     'Top-k Annualized Return': top_k_metrics.loc[10, 'Annualized Return'],
                     'Top-k Annualized Sharpe': top_k_metrics.loc[10, 'Annualized Sharpe'],
                     'Top-k Annualized Sortino': top_k_metrics.loc[10, 'Annualized Sortino'],
                     'Model Configs': str(model.get_params()),
                     'Period Range': period_range,
                     'Study Period Start Date': start_date.isoformat(),
                     'Study Period End Date': end_date.isoformat()
                     }

    logger = CSVWriter(output_path=os.path.join(ROOT_DIR, 'data', 'training_log.csv'),
                       field_names=list(data_record.keys()))
    logger.add_line(data_record)

    external_file_path = r'E:\OneDrive - student.kit.edu\[01] Studium\[06] Seminare\[04] Electronic Markets & User Behavior\[04] Live Data'
    if os.path.exists(external_file_path):
        logger = CSVWriter(output_path=os.path.join(external_file_path, 'training_log.csv'),
                           field_names=list(data_record.keys()))
        logger.add_line(data_record)
        add_to_json(data_record_json, os.path.join(external_file_path, 'training_log.json'))

    add_to_json(data_record_json, 'data/training_log.json')

    # JOB: RandomForest experiment logging
    # logger = CSVWriter(output_path=os.path.join(ROOT_DIR, 'data', 'rf_training_log.csv'),
    #                    field_names=list(forest_record.keys()))
    # logger.add_line(forest_record)

    print('Done testing on unseen data.')
    timer.stop()

    return full_portfolio['norm_prediction'].values


# noinspection DuplicatedCode
def get_market_metrics(market_portfolio: pd.DataFrame, t_costs: float) -> Tuple[pd.Series, pd.Series]:
    """
    Get performance metrics for full market portfolio

    :param t_costs: Transaction costs per half-turn
    :param market_portfolio: DataFrame including full test set (market portfolio)
    :return: Tuple of market portfolio metrics (Series) and cumulative returns series (Series)
    """

    market_portfolio_metrics = pd.Series([]).rename('Market')
    market_portfolio_metrics.index.name = 'Metrics'

    return_series = market_portfolio.loc[:, 'daily_return'].groupby(level=['datadate']).mean()
    cumulative_return = (return_series + 1).cumprod().rename('Cumulative Market Return')
    cumulative_return.index.name = 'Time'
    # cumulative_return.plot(title='Cumulative Market Performance')
    # plt.legend(loc='best')
    # plt.show()

    # JOB: Calculate metrics
    # noinspection DuplicatedCode
    annualized_sharpe = calc_sharpe(market_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean(),
                                    annualize=True)
    annualized_sharpe_atc = calc_sharpe(
        market_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean() - 4 * t_costs,
        annualize=True)

    annualized_sortino = calc_sortino(market_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean(),
                                      annualize=True)
    annualized_sortino_atc = calc_sortino(
        market_portfolio.loc[:, ['daily_return']].groupby(level=['datadate']).mean() - 4 * t_costs,
        annualize=True)

    mean_daily_return = market_portfolio.groupby(level=['datadate'])['daily_return'].mean().mean()
    mean_daily_excess_return = calc_excess_returns(
        market_portfolio.groupby(level=['datadate'])['daily_return'].mean().rename('daily_return')).mean()

    market_portfolio_metrics.loc['Mean Daily Return'] = mean_daily_return
    market_portfolio_metrics.loc['Annualized Return'] = annualize_metric(mean_daily_return)
    market_portfolio_metrics.loc['Mean Daily Excess Return'] = mean_daily_excess_return
    market_portfolio_metrics.loc['Annualized Excess Return'] = annualize_metric(mean_daily_excess_return)
    market_portfolio_metrics.loc['Annualized Sharpe'] = annualized_sharpe
    market_portfolio_metrics.loc['Annualized Sortino'] = annualized_sortino

    # JOB: Add metrics incl. transaction costs of 5 bps per half-turn
    market_portfolio_metrics.loc['Mean Daily Return_atc'] = mean_daily_return - 4 * t_costs
    market_portfolio_metrics.loc['Annualized Return_atc'] = annualize_metric(mean_daily_return - 4 * t_costs)
    market_portfolio_metrics.loc['Mean Daily Excess Return_atc'] = mean_daily_excess_return - 4 * t_costs
    market_portfolio_metrics.loc['Annualized Excess Return_atc'] = annualize_metric(
        mean_daily_excess_return - 4 * t_costs)
    market_portfolio_metrics.loc['Annualized Sharpe_atc'] = annualized_sharpe_atc
    market_portfolio_metrics.loc['Annualized Sortino_atc'] = annualized_sortino_atc

    return market_portfolio_metrics, cumulative_return


if __name__ == '__main__':
    print(f'{Fore.YELLOW}{Style.BRIGHT}This module has no action on execution as \'main\'.{Style.RESET_ALL}')
