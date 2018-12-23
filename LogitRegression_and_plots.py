import pandas as pd
import numpy as np
import statsmodels.api as sm
import scikitplot as skplt
import matplotlib.pyplot as plt
from dataPreprocessing import data_preprocessing
from utils import standardize_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def perform_logit(df, dv, ivs):
    logit = sm.Logit(df[dv], df[ivs])
    result = logit.fit()
    return result


def descriptive_analysis_of_logit(logit_result, df, dv, ivs):
    print(logit_result.summary2())
    prediction = np.array([1-logit_result.predict(), logit_result.predict()])
    print(df.describe())
    #lift curve
    skplt.metrics.plot_lift_curve(df[dv], prediction.T)
    plt.show()
    skplt.metrics.plot_ks_statistic(df[dv], prediction.T)
    plt.show()
    skplt.metrics.plot_cumulative_gain(df[dv], prediction.T)
    plt.show()
    skplt.metrics.plot_roc_curve(df[dv], prediction.T)
    plt.show()
    skplt.metrics.plot_confusion_matrix(df[dv], logit_result.predict() > 0.5)
    plt.show()





#Loading and preparing the data 
df = pd.read_csv("TelcoCustomerChurn.csv")
df = data_preprocessing(df)

# Filtering and fitting variables that should be included
logit_dv = 'Churn_Yes'
# adding an intercept
df['intercept'] = 1.0
df["tenure2"] = df["tenure"] ** 2
df["MonthlyCharges2"] = df["MonthlyCharges"] ** 2
df["MonthlyCharges3"] = df["MonthlyCharges"] ** 3

# selecting desired columns
logit_ivs = [
 'tenure',
 'tenure2',
 'MonthlyCharges3',
 'SeniorCitizen_Yes',
 'MonthlyCharges',
 'PaperlessBilling_Yes',
 'OnlineSecurity_Yes',
 'TechSupport_Yes',
 'Contract_Month-to-month',
 'PaymentMethod_Electronic check',
 'intercept'
 ]

X_train, X_test, y_train, y_test = train_test_split(df[logit_ivs], df[logit_dv], test_size=0.30, random_state=42)
logit = sm.Logit(y_train, X_train)
logit_result = logit.fit()

print(logit_result.summary2())
prediction = np.array([1-logit_result.predict(), logit_result.predict()])
print(df.describe())
#lift curve
skplt.metrics.plot_lift_curve(y_train, prediction.T)
plt.show()
skplt.metrics.plot_ks_statistic(y_train, prediction.T)
plt.show()
skplt.metrics.plot_cumulative_gain(y_train, prediction.T)
plt.show()
skplt.metrics.plot_roc_curve(y_train, prediction.T)
plt.show()
skplt.metrics.plot_confusion_matrix(y_train, logit_result.predict() > 0.5)
plt.show()


#logit_result = perform_logit(df, logit_dv, logit_ivs)
#descriptive_analysis_of_logit(logit_result, df, logit_dv, logit_ivs)