'''
library doc string
author: Esra
date: 29.11.2022
'''

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import os
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(file_path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df=pd.read_csv(file_path)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.savefig('./images/eda/histogram.png')
    
    plt.figure(figsize=(20,10))
    sns.heatmap( df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/heatmap.png')
    
    
def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col_val in category_lst:
        category_groups=df.groupby(col_val).mean()['Churn']
        cat_list=[]
        for val in df[col_val]: 
            cat_list.append(category_groups.loc[val])
        df[col_val+'_Churn']=cat_list
    return  df.columns

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    y=df[response]
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test
    
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
        # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    
    
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("randomforest_results.png")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("logistics_results.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    

        

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    #feature_importance_plot(lrc, X_test, y_test)
    feature_importance_plot(cv_rfc.best_estimator_, X_test, y_test)
   
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    
    #ROC curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/Roc_Curves.jpg")
    plt.close()
    
    #explainer cv_rfc    
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

if __name__ == "__main__":
    file_path= r"./data/bank_data.csv"
    df=import_data(file_path)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df)
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category' ]

    quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]
    
    encoder_helper(df, cat_columns, 'Churn') 
    X_train, X_test, y_train, y_test=perform_feature_engineering(df, 'Churn')
    train_models(X_train, X_test, y_train, y_test) 