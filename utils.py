import pandas as pd


def load_data(path):
    path = "TelcoCustomerChurn.csv"
    df_telco = pd.read_csv(path)
    return df_telco
