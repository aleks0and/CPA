import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data
from dataPreprocessing import data_preprocessing


def frequency_measure_visualized(df):
    plt.subplots_adjust(wspace=2, hspace=2)
    fig1 = plt.figure(figsize=(10, 15))
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
    plt.show(fig1)



path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
frequency_measure_visualized(df)