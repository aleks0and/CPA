import pandas as pd
import numpy as np
from dataPreprocessing import data_preprocessing
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from functools import reduce

# Defining the bootstrap function
def logit_bootstrap(df,dv,ivs,size=1):
    inds = np.arange(len(df))

    y = df[dv]
    x = df[ivs]

    # Initialize replicates: bs_logit_results
    bs_logit_results = []
    bs_logit_accuracy = []
    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x = pd.DataFrame(columns=ivs)
        for name in ivs:
            bs_x[name] =(x[name][bs_inds])
        bs_y = y[bs_inds]
        logit = sm.Logit(bs_y, bs_x)
        logit_result = logit.fit(disp = 0)
        bs_logit_results.append(logit_result.params)
        logRegress = LogisticRegression(solver='liblinear')
        logRegress.fit(bs_x, bs_y)
        bs_logit_accuracy.append(logRegress.score(bs_x, bs_y))
    return bs_logit_results, bs_logit_accuracy


# ============================================TESTING=======================================================
# df = pd.read_csv("TelcoCustomerChurn.csv")
# df = data_preprocessing(df)
#
#
# logit_dv = 'Churn_Yes'
# # adding an intercept
# df['intercept'] = 1.0
# df["tenure2"] = df["tenure"] ** 2
# # selecting desired columns
# logit_ivs = [
#  'tenure',
#  'tenure2',
#  'MonthlyCharges',
#  'SeniorCitizen_Yes',
#  'InternetService_Fiber optic',
#  'Contract_Month-to-month',
#  'PaymentMethod_Electronic check',
#  'intercept'
#  ]
#
# bs_result, bs_accuracy = logit_bootstrap(df, logit_dv, logit_ivs, 50)
# print(len(bs_result[1]))
# print(reduce(lambda x, y: x + y, bs_accuracy) / float(len(bs_accuracy)))
