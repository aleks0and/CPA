import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data
from dataPreprocessing import data_preprocessing, finding_outliers_for_columns_list

#this has to be corrected as we dont want to be manually adding senior citizen.
def frequency_measure_visualized(df, names_to_drop):
    names_to_drop = set(names_to_drop)
    dummies_names = [x for x in list(df) if x not in names_to_drop]
    plt.subplots_adjust(wspace=2, hspace=2)
    plt.figure(figsize=(10, 15))
    for i in range(len(dummies_names)):
        size = df[dummies_names[i]].value_counts()
        plt.subplot(7, 3, i + 1)
        plt.pie(size, labels=["no", "yes"], shadow=True, autopct='%1.0f%%')
        plt.title(dummies_names[i])
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



path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
names = ['tenure', 'MonthlyCharges', 'TotalCharges']
names_to_drop = names
names.append('customerID')
frequency_measure_visualized(df, names_to_drop)
plotting_histograms_for_column_list(names, df)
plotting_boxplots_for_column_list(names, df)
df_without_outliers = finding_outliers_for_columns_list(names, df)
