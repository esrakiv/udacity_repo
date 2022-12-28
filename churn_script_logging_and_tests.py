'''
library doc string
author: Esra
date: 29.11.2022

Test script of the churn_library.py
'''
import os
import logging
#import churn_library_solution as cls
import pandas as pd
import churn_library as cl
import constants as ct

logging.basicConfig(
    filename='./logs/churn_library_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
    '''
    try:
        df_data = import_data(ct.FILE_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda,data_df):
    '''
    test perform eda function
    '''
    perform_eda(data_df)
    try:
        assert os.path.isfile('./images/eda/histogram.png')
        logging.info("Import histogram: SUCCESS")
        assert os.path.isfile('./images/eda/heatmap.png')
        logging.info("Import heatmap: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err


def test_encoder_helper(df_data, cat_cols):
    '''
    test encoder_helper function
    '''
    try:
        for col_val in cat_cols:
            assert df_data[col_val].dtype.name == 'object'
        logging.info("These columns are categoric: SUCCESS")
        for new_col in df_data.columns:
            if 'Churn' in new_col:
                assert df_data[new_col].dtype != 'object'
        logging.info("New columns exists and numeric: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_encoder_helper: FAIL")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df_data):
    '''
    test perform_feature_engineering
    '''
    try:
        # train test split
        assert perform_feature_engineering(df_data, 'Churn')
        logging.info("split data as train and test: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_perform_feature_engineering: FAIL")
        raise err


def test_train_models(
        train_models,
        dataset_x_train,
        dataset_x_test,
        dataset_y_train,
        dataset_y_test):
    '''
    test train_models
    '''
    try:
        train_models(
            dataset_x_train,
            dataset_x_test,
            dataset_y_train,
            dataset_y_test)
        assert os.path.isfile('./images/sharp_values.png')
        logging.info("train model: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train model: FAIL")
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    df_bank_data = cl.import_data(ct.FILE_PATH)
    df_bank_data[ct.RESPONSE] = df_bank_data['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    test_eda(cl.perform_eda, df_bank_data)
    test_encoder_helper(df_bank_data, ct.cat_columns)
    target_y = df_bank_data[ct.RESPONSE]
    dataset_x = pd.DataFrame()
    cl.encoder_helper(df_bank_data, ct.cat_columns)
    cl.perform_feature_engineering(df_bank_data, ct.RESPONSE)
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df_bank_data, ct.RESPONSE)
    test_perform_feature_engineering(cl.perform_feature_engineering, df_bank_data)
    test_train_models(cl.train_models, x_train, x_test, y_train, y_test)
