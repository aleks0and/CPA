#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:37:46 2018

@author: Mac
"""

import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

#Loading and preparing the data 
df = pd.read_csv("TelcoCustomerChurn.csv")
df.head()

df = data_preprocessing(df)
df = df.drop('customerID', axis=1)
columns_ = df.columns.tolist()
stan_df = standardize_data(df, standardization = True)
stan_df.columns = [columns_]
descriptives = df.describe()

# Filtering and fitting variables that should be included
current_var = df.columns.tolist()
logit_var = [
 'Churn_Yes',
 'tenure',
 'SeniorCitizen_Yes',
 'MonthlyCharges',
 'TotalCharges',
 'gender_Male',
 'Dependents_Yes',
 'PaperlessBilling_Yes',
 'OnlineSecurity_Yes',
 'DeviceProtection_Yes',
 'TechSupport_Yes',
 'InternetService_DSL',
 'Contract_Month-to-month',
 'PaymentMethod_Electronic check'
 ]
df = df[logit_var]
# adding an intercept 
df['intercept'] = 1.0

# Fitting the regression
ind_var = df.columns[1:]
logit = sm.Logit(df['Churn_Yes'], df[ind_var])
result = logit.fit()
result.summary2()
    
# Plotting evaluation statistics
import scikitplot as skplt
pred = np.array([1-result.predict(), result.predict()])
skplt.metrics.plot_lift_curve(df["Churn_Yes"], pred.T)
skplt.metrics.plot_ks_statistic(df["Churn_Yes"], pred.T)
skplt.metrics.plot_roc_curve(df["Churn_Yes"], pred.T)
skplt.metrics.plot_confusion_matrix(df["Churn_Yes"], result.predict() > 0.5)