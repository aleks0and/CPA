import pandas as pd
import numpy as np
from dataPreprocessing import data_preprocessing
import statsmodels.api as sm


# Defining the bootstrap function
def logit_bootstrap(df,dv,ivs,size=1):
    inds = np.arange(len(df))

    y = df[dv]
    x = df[ivs]

    # Initialize replicates: bs_logit_results
    bs_logit_results = []

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x = []
        for j in range(len(ivs)):
            bs_x.append(x[ivs[j]][bs_inds])
        bs_y = y[bs_inds]
        logit = sm.Logit(bs_y, bs_x)
        logit_result = logit.fit()
        bs_logit_results.append(logit_result)

    return bs_logit_results



df = pd.read_csv("TelcoCustomerChurn.csv")
df = data_preprocessing(df)


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

logit_bootstrap(df,logit_dv,logit_ivs,1000)