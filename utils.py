import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
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
