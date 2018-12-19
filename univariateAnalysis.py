import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data
from dataPreprocessing import data_preprocessing

#this has to be corrected as we dont want to be manually adding senior citizen.
def frequency_measure_visualized(df):
    plt.subplots_adjust(wspace=2, hspace=2)
    plt.figure(figsize=(10, 15))
    store = 0
    for i in range(len(df.columns) - 3):
        if i == 0 | i > 4:
            size = df[df.columns[i]].value_counts()
            plt.subplot(7, 3, i - 4)
            plt.pie(size, labels=["no", "yes"], shadow=True, autopct='%1.0f%%')
            plt.title(df.columns[i])
            store = i
    size = df["SeniorCitizen_Yes"].value_counts()
    plt.subplot(7, 3, store - 3)
    plt.pie(size, labels=["no", "yes"], shadow=True, autopct='%1.0f%%')
    plt.title("SeniorCitizen_Yes")
    plt.show()


def plotting_histograms_for_column_list(names_list, df):
    i = 0
    plt.figure(figsize=(10,15))
    plt.subplots_adjust(wspace=1, hspace=1)
    for name in names_list:
        plt.subplot(3, len(names_list)/3, i+1)
        plt.hist(df[name])
        plt.title(name)
        i += 1
    plt.show()


def plotting_boxplots_for_column_list(names_list, df):
    i = 0
    plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=1, hspace=1)
    for name in names_list:
        plt.subplot(3, len(names_list)/3, i+1)
        plt.boxplot(df[name])
        plt.title(name)
        i += 1
    plt.show()


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


path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
print(list(df))
frequency_measure_visualized(df)
names = ['tenure', 'MonthlyCharges', 'TotalCharges']
plotting_histograms_for_column_list(names, df)
plotting_boxplots_for_column_list(names, df)
df_without_outliers = finding_outliers_for_columns_list(names, df)
