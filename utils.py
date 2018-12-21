import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from dataPreprocessing import data_preprocessing

def load_data(path):
    path = "TelcoCustomerChurn.csv"
    df_telco = pd.read_csv(path)
    return df_telco


def standardize_data(data, standardization):
    result = data.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result


def standardize_columns(data, standardization, column_names):
    if standardization:
        for name in column_names:
            std_dev = data[name].std(axis=0)
            mean = data[name].mean(axis=0)
            # done for checking
            # print(std_dev)
            # print(mean)
            data[name].apply(lambda x: (x-mean) / std_dev)
        result = data.values
    else:
        result = data.values
    return result


def plot_clusters(data, predicted_clusters, initialized_kmeans, number_of_clusters):
    for i in range(0, number_of_clusters):
        color = cm.nipy_spectral(float(i) / number_of_clusters)
        plt.scatter(data[predicted_clusters == i, 0],
                    data[predicted_clusters == i, 1],
                    s=50, c=color,
                    marker='o', edgecolor=color,
                    label='cluster %d' % (i+1))
    color = cm.nipy_spectral(float(number_of_clusters) / number_of_clusters)
    plt.scatter(initialized_kmeans.cluster_centers_[:, 0],
                initialized_kmeans.cluster_centers_[:,  1],
                s=250, marker='*',
                c=color, edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


# path = "TelcoCustomerChurn.csv"
# df_telco = pd.read_csv(path)
# df_preprocessed = data_preprocessing(df_telco)
# columns_to_standardize = ['tenure', 'MonthlyCharges', 'TotalCharges']
# df_preprocessed = standardize_columns(df_preprocessed, True, columns_to_standardize)