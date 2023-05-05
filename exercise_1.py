# %%
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

tracks = pd.read_csv("./fma_metadata/tracks.csv", index_col=0, header=[0, 1])

tracks.shape

tracks = tracks.loc[tracks['set','subset'] == 'small'].reset_index(drop=True)
tracks

tracks.shape

tracks.shape[0]

features = pd.read_csv("./fma_metadata/features.csv", encoding='utf-8-sig')
features

first_row = features.iloc[0, :]
second_row = features.iloc[1, :]
third_row = features.iloc[2, :]

new_column_names = [third_row.iloc[0]] + list(first_row.iloc[1:].values)

features.columns = new_column_names
features = features.iloc[3:, :].reset_index(drop=True)
features

ids = list(set(tracks.index.to_list()))
features_sm = features.loc[ids]
features_sm = features_sm.drop(['track_id'], axis=1)
features_sm

print(features_sm.shape)

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def cluster_metrics(data, labels):
    clusters = np.unique(labels)
    radii = []
    diameters = []
    densities_r = []
    densities_d = []

    for cluster in clusters:
        cluster_data = data[labels == cluster]
        centroid = np.mean(cluster_data, axis=0)
        intra_distances = pairwise_distances(cluster_data)
        radius = np.max(np.linalg.norm(cluster_data - centroid, axis=1))
        diameter = np.max(intra_distances)
        density_r = None
        if radius == 0:
            density_r = 0
        else:
            density_r = len(cluster_data) / (radius ** 2)
        density_d = None
        if diameter == 0:
            density_d = 0
        else:
            density_d = len(cluster_data) / (diameter ** 2)
        

        radii.append(radius)
        diameters.append(diameter)
        densities_r.append(density_r)
        densities_d.append(density_d)

    return radii, diameters, densities_r, densities_d

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = features_sm.values
X = StandardScaler().fit(X).transform(X)
df_numeric_scaled = scaler.fit_transform(features_sm.values)

results = []
for k in range(8, 17):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(df_numeric_scaled)
    
    radii, diameters, density_r, density_d = cluster_metrics(df_numeric_scaled, labels)
    
    result = {
        'k': k,
        'radii': radii,
        'diameters': diameters,
        'density_r': density_r,
        'density_d': density_d,
    }
    results.append(result)

for result in results:
    print(f"Number of clusters (k): {result['k']}")
    print(f"Radius: {result['radii']}")
    print(f"Diameters: {result['diameters']}")
    print(f"Densities_r: {result['density_r']}")
    print(f"Densities_d: {result['density_d']}")
    print()

# %% [markdown]
# #### Escolhemos k=9, porque após fazermos uma analise de todos os valores de k, este foi o que nos apresentou um menor raio e diametro e com valores de densidade maiores.

from sklearn.cluster import KMeans

def bfr_clustering_v2(data, n_clusters, init='k-means++', random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
    kmeans.fit(data)
    cluster_centers_ = kmeans.cluster_centers_
    labels_ = kmeans.labels_

    # Initialize cluster summary statistics
    cluster_stats_ = []
    for i in range(n_clusters):
        cluster_data = data[labels_ == i]
        n = len(cluster_data)
        cluster_sum = cluster_data.sum(axis=0)
        cluster_sum_sq = (cluster_data ** 2).sum(axis=0)
        cluster_stats_.append({'n': n, 'sum': cluster_sum, 'sum_sq': cluster_sum_sq})

    return labels_, cluster_centers_

data = pd.read_csv('./fma_metadata/features.csv', header=[0, 1, 2], index_col=0)

data.columns = ['_'.join(col).strip() for col in data.columns.values]

#drop any missing values
data = data.dropna()

#cluster the dataset using the bfr_clustering_v2 function with k=9
labels, cluster_centers = bfr_clustering_v2(data, n_clusters=9, init='k-means++', random_state=42)

print("Cluster assignments:", labels)
print("Cluster centers:", cluster_centers)


from scipy.spatial.distance import mahalanobis

def calculate_inverse_covariance_matrices(data, labels, cluster_centers):
    inverse_cov_matrices = []
    for i in range(cluster_centers.shape[0]):
        cluster_data = data[labels == i]
        cov_matrix = np.cov(cluster_data.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        inverse_cov_matrices.append(inv_cov_matrix)
    return inverse_cov_matrices

inverse_cov_matrices = calculate_inverse_covariance_matrices(data, labels, cluster_centers)

#calculate the Mahalanobis distances for each data point to its assigned centroid
mahalanobis_distances = np.zeros(data.shape[0])
for i in range(data.shape[0]):
    cluster_idx = labels[i]
    centroid = cluster_centers[cluster_idx]
    inv_cov_matrix = inverse_cov_matrices[cluster_idx]
    mahalanobis_distances[i] = mahalanobis(data.iloc[i], centroid, inv_cov_matrix)

print("Mahalanobis distances:", mahalanobis_distances)

#function to calculate the average Mahalanobis distance and standard deviation for each cluster
def calculate_cluster_mahalanobis_stats(labels, mahalanobis_distances, n_clusters):
    cluster_stats = []
    for i in range(n_clusters):
        cluster_distances = mahalanobis_distances[labels == i]
        mean_distance = np.mean(cluster_distances)
        std_distance = np.std(cluster_distances)
        cluster_stats.append({'mean': mean_distance, 'std': std_distance})
    return cluster_stats

std_multiplier = 2

cluster_stats = calculate_cluster_mahalanobis_stats(labels, mahalanobis_distances, 9)

new_labels = np.zeros_like(labels)

# Assign each point to a cluster or mark it as unassigned (-1) based on the threshold
for i in range(data.shape[0]):
    cluster_idx = labels[i]
    threshold = cluster_stats[cluster_idx]['mean'] + std_multiplier * cluster_stats[cluster_idx]['std']
    if mahalanobis_distances[i] <= threshold:
        new_labels[i] = cluster_idx
    else:
        new_labels[i] = -1

print("New cluster assignments:", new_labels)

from collections import defaultdict

#function to create a dictionary with the keys being the clusters and the values being lists of all the data points belonging to each cluster
def create_cluster_dictionary(data, labels):
    cluster_dict = defaultdict(list)
    for i in range(data.shape[0]):
        cluster_idx = labels[i]
        cluster_dict[cluster_idx].append(data.iloc[i].values)
    return cluster_dict

cluster_dictionary = create_cluster_dictionary(data, new_labels)

print("Cluster dictionary:")
for key, value in cluster_dictionary.items():
    print(f"Cluster {key}: {len(value)} points")

import pandas as pd
import matplotlib.pyplot as plt

tracks = pd.read_csv('./fma_metadata/tracks.csv', header=[0, 1], index_col=0)

tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]

#extract the genre information for each track
genres = tracks['track_genre_top']

#map the genre information to the cluster assignments
data_genres = data.assign(genre=genres).dropna(subset=['genre'])

#update new_labels' indices to match data_genres
new_labels_aligned = pd.Series(new_labels, index=data.index)

cluster_genres = data_genres.loc[:, 'genre'].groupby(new_labels_aligned).agg(list)

#function to calculate the genre count for each cluster
def calculate_genre_count(cluster_genres):
    genre_counts = []
    for i, genres in cluster_genres.items():
        genre_count = pd.Series(genres).value_counts()
        genre_counts.append(genre_count)
    return genre_counts

#calculate the genre count for each cluster
genre_counts = calculate_genre_count(cluster_genres)

#create a bar chart to visualize the total count of music genres in each cluster
def plot_genre_counts(genre_counts):
    fig, axes = plt.subplots(nrows=len(genre_counts), ncols=1, figsize=(10, 5 * len(genre_counts)))

    for i, ax in enumerate(axes):
        genre_counts[i].plot(kind='bar', ax=ax)
        ax.set_title(f'Cluster {i}')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

plot_genre_counts(genre_counts)



