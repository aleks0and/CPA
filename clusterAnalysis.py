import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from utils import load_data, standardize_data
from dataPreprocessing import data_preprocessing


def hierarchical_cluster_analysis(df):
    plt.figure(figsize=(10, 15))
    clusters = linkage(pdist(df, metric='euclidean'), method='complete')
    labels = ['cluster 1', 'cluster 2', 'coefficients', 'no. of items in clust.']
    clusters_labeled = pd.DataFrame(clusters, columns=labels,
                                    index=['stage %d' % (i + 1) for i in range(clusters.shape[0])]
                                    )
    clusterDendogram = dendrogram(clusters, color_threshold=712, no_labels = True)
    plt.title("Dendogram")
    plt.xlabel("Cluster Labels excluded for better comprehension")
    plt.ylabel("Dist")
    plt.tight_layout()
    plt.show()


def k_means_analysis_with_silhouette_plotting(df, number_of_clusters):
    km = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    # To me it looks like clusters should be either 3 or 6
    y_km = km.fit_predict(df)
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]

    # Creating the silhouette plot
    silhouette_vals = silhouette_samples(df, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    plt.figure(figsize=(10, 15))
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.show()


path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
# we are dropping id, gender and age variables as they should not be included in cluster analysis
columns_to_drop = ['customerID', 'SeniorCitizen_Yes', 'gender_Male']
df = df.drop(columns_to_drop, axis=1)
df_standardized = standardize_data(df, True)
hierarchical_cluster_analysis(df_standardized)
number_of_clusters = 6
k_means_analysis_with_silhouette_plotting(df_standardized, number_of_clusters)
