import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    path = "TelcoCustomerChurn.csv"
    df_telco = pd.read_csv(path)
    return df_telco


def standardize_data(data, standardization):
    result = data.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result
