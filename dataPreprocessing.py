import pandas as pd


def data_preprocessing(df):
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
    df = df.dropna()
    return df


#path = "TelcoCustomerChurn.csv"
#df_telco = pd.read_csv(path)
#df_preprocessed = data_preprocessing(df_telco)
