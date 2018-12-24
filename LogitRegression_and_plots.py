import pandas as pd
import numpy as np
import statsmodels.api as sm
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
from dataPreprocessing import data_preprocessing
from utils import standardize_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



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

# this function needs to be updated with the test set
def descriptive_analysis_of_logit_given_dataset(logit_result, data, df):
    print(logit_result.summary2())
    prediction = np.array([1-logit_result.predict(), logit_result.predict()])
    print(df.describe())
    #lift curve
    skplt.metrics.plot_lift_curve(data, prediction.T)
    plt.show()
    skplt.metrics.plot_ks_statistic(data, prediction.T)
    plt.show()
    skplt.metrics.plot_cumulative_gain(data, prediction.T)
    plt.show()
    skplt.metrics.plot_roc_curve(data, prediction.T)
    plt.show()
    skplt.metrics.plot_confusion_matrix(data, logit_result.predict() > 0.5)
    plt.show()




#Loading and preparing the data
df = pd.read_csv("TelcoCustomerChurn.csv")
df = data_preprocessing(df, standardize=True)
 
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
  'MonthlyCharges',
  'SeniorCitizen_Yes',
  'InternetService_Fiber optic',
  'Contract_Month-to-month',
  'PaymentMethod_Electronic check',
  'intercept'
  ]
sns.heatmap(df[logit_ivs].corr())
plt.show()
x_train, x_test, y_train, y_test = train_test_split(df[logit_ivs], df[logit_dv], test_size=0.30, random_state=42)
#=========================================ACCURACY OF THE MODEL===================================================
logRegress = LogisticRegression()
logRegress.fit(x_train, y_train)
accuracy = logRegress.score(x_train, y_train)
#=================================================================================================================
print(accuracy)
logit = sm.Logit(y_train, x_train)
logit_result = logit.fit()

print(logit_result.summary2())
 
prediction = np.array([1-logit_result.predict(x_test), logit_result.predict(x_test)])
print(df.describe())
#lift curve
skplt.metrics.plot_lift_curve(y_test, prediction.T)
plt.show()
skplt.metrics.plot_ks_statistic(y_test, prediction.T)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, prediction.T)
plt.show()
skplt.metrics.plot_roc_curve(y_test, prediction.T)
plt.show()
skplt.metrics.plot_confusion_matrix(y_test, logit_result.predict(x_test) > 0.5)
plt.show()


#logit_result = perform_logit(df, logit_dv, logit_ivs)
#descriptive_analysis_of_logit(logit_result, df, logit_dv, logit_ivs)

# gamma_model = sm.GLM(y_train, x_train, family=sm.families.Gamma())
# gamma_results = gamma_model.fit()
# print(gamma_results.summary())