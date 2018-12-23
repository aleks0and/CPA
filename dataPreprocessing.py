import pandas as pd
from utils import standardize_data


def data_preprocessing(df, standardize = False):

    # specifying which columns contain dummy variables
    dummies_names = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn',
                     'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies']
    df = pd.get_dummies(df, columns=dummies_names, drop_first=True)
    # dropping not needed columns
    columns_to_drop = ['MultipleLines_No phone service', 'OnlineSecurity_No internet service',
                       'OnlineBackup_No internet service', 'DeviceProtection_No internet service',
                       'TechSupport_No internet service', 'StreamingTV_No internet service',
                       'StreamingMovies_No internet service']
    df = df.drop(columns_to_drop, axis=1)
    # declaring dummies without dropping the first dummy for each
    dummies_names = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=dummies_names)
    # dropping not needed dummies
    columns_to_drop = ['InternetService_No', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)']
    df = df.drop(columns_to_drop, axis=1)
    # renaming Senior Citizen column to indicate their dummy status
    senior_citizen = 'SeniorCitizen'
    df = df.rename(index=str, columns={senior_citizen: senior_citizen+"_Yes"})
    # converting values from string to numeric values
    total_charges = 'TotalCharges'
    df[total_charges] = pd.to_numeric(df[total_charges], errors='coerce')
    

    # dropping missing values
    # only missing values that are in the dataset are in TotalCharges columns,
    # which we do not use, so we are not dropping any missing values
    # df = df.dropna()

    if standardize:
        df = standardize_data(df, standardization = True, column_names = ['tenure', 'MonthlyCharges', 'TotalCharges'])

    return df


def finding_outliers_for_columns_list(names, df):
    """ the result is the combined dataset without the outliers from the columns given in the given order"""
    filter = df
    for name in names:
        Q1 = df[name].quantile(0.25)
        Q3 = df[name].quantile(0.75)
        IQR = Q3 - Q1
        print(str(len(filter)) + " observations BEFORE removing outliers")
        filter = filter.query('(@Q1 - 1.5 * @IQR) <= ' + name + '<= (@Q3 + 1.5 * @IQR)')
        print(str(len(filter)) + " observations AFTER removing outliers from " + name + " column")
    return filter



# path = "TelcoCustomerChurn.csv"
# df_telco = pd.read_csv(path)
# df_preprocessed = data_preprocessing(df_telco)
