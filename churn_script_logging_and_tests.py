import os
import logging
#import churn_library_solution as cls
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library_results.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        plt.savefig('./images/eda/histogram.png')
        logging.info("Import histogram: SUCCESS")
        plt.savefig('./images/eda/heatmap.png')
        logging.info("Import heatmap: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
    raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        total_words = len(df.columns.split())
        assert type(df.columns, 'str')
        logging.info("Testing num_words: SUCCESS")
        return total_words
    except AssertionError as err:
        logging.error("Testing test_encoder_helper: FAIL")
        raise err   


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        assert train_test_split(X, y, test_size= 0.3, random_state=42)
        logging.info("Testing num_words: SUCCESS")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error("Testing test_perform_feature_engineering: FAIL")
        raise err   

def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        assert plt.savefig("images/results/Roc_Curves.jpg")
        logging.info("save Roc_Curves: SUCCESS")
        assert joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info("save logistic model: SUCCESS")
        assert shap_values == explainer.shap_values(X_test)
        logging.info("sharp values were created: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train model: FAIL")
        raise err    
    


if __name__ == "__main__":
    pass








