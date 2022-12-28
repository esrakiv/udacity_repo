'''
library doc string
author: Esra
date: 29.11.2022
This library consists of functions which gives a churn model from a csv dataset 
'''
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as ct
; sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def import_data(file_path):
    '''
    By given file_path, function returns dataframe for the csv found at pth
    '''
    data_df = pd.read_csv(file_path)
    return data_df

def perform_eda(data_df):
    '''
    perform eda on df and save figures to images folder
    '''
    plt.figure(figsize=(20, 10))
    data_df['Churn'].hist()
    plt.savefig('./images/eda/histogram_churn.png')
    plt.close()
    
    plt.figure(figsize=(20,10)) 
    data_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/histogram_marital_status.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap_all_features_correlation.png')
    plt.close()


def encoder_helper(data_df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category
    '''
    for col_val in category_lst:
        category_groups = data_df.groupby(col_val).mean()['Churn']
        cat_list = []
        for val in data_df[col_val]:
            cat_list.append(category_groups.loc[val])
        data_df[col_val + '_Churn'] = cat_list
    return data_df.columns


def perform_feature_engineering(data_df, response):
    '''
    dataframe is split train/test datasets
    '''
    target_df = data_df[response]
    dataset_x = pd.DataFrame()
    dataset_x[ct.keep_cols] = data_df[ct.keep_cols]
    # train test split
    dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = train_test_split(dataset_x, target_df, test_size=0.3, random_state=42)
    return dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test

def classification_report_image(dataset_y_train, dataset_y_test, y_train_preds_lr, y_train_preds_rf, 
                                y_test_preds_lr, y_test_preds_rf): 
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder
    '''
    plt.rc('figure', figsize=(10, 10)) 
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report( dataset_y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report( dataset_y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off') 
    plt.savefig("./images/randomforest_results.png", dpi=200, format='png', bbox_inches='tight')
    plt.close()

    plt.rc('figure', figsize=(10, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(dataset_y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(dataset_y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/logistics_results.png",dpi=200, format='png', bbox_inches='tight')
    plt.close()
     


def feature_importance_plot(model,dataset_x):
    '''
    By given model and dataset, it creates and stores the feature importances in pth
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [dataset_x.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(dataset_x.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(dataset_x.shape[1]), names, rotation=90)
    
    plt.savefig("./images/results/feature_importance.png")
    plt.close()


def train_models(dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test):
    '''
    by feature engineering output as inputs, train and store model results
    '''
    rfc = RandomForestClassifier(random_state=42) 
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(dataset_x_train, dataset_y_train)

    lrc.fit(dataset_x_train, dataset_y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(dataset_x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(dataset_x_test)

    y_train_preds_lr = lrc.predict(dataset_x_train)
    y_test_preds_lr = lrc.predict(dataset_x_test)

    #feature_importance_plot(lrc, X_test, y_test)
    feature_importance_plot(cv_rfc.best_estimator_, dataset_x_test)

    classification_report_image(dataset_y_train, dataset_y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    
    plt.figure()
    plt.title("best estimator") 
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(dataset_x_test)
    shap.summary_plot(shap_values, dataset_x_test, plot_type="bar", show=False)
    plt.savefig('./images/sharp_values.png')
    plt.close()

    # ROC curves
    lrc_plot = plot_roc_curve(lrc, dataset_x_test, dataset_y_test)

    # plots
    plt.figure(figsize=(15, 8))
    a_x= plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, dataset_x_test, dataset_y_test, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig("images/results/Roc_Curves.jpg")
    plt.close()


if __name__ == "__main__": 
    df_bank_data = import_data(ct.FILE_PATH)
    df_bank_data[ct.RESPONSE] = df_bank_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df_bank_data)
    encoder_helper(df_bank_data, ct.cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_bank_data, ct.RESPONSE)
    train_models(X_train, X_test, y_train, y_test)
