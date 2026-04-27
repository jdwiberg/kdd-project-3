import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, silhouette_score, pairwise_distances, homogeneity_score, \
    v_measure_score, completeness_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from main import preprocessing

# CS 4445 - Project 3
# Segment: Hierarchical Clustering
# Group: Shane G. , Jake W. , Victor C.
# April 23rd 2026

# Preprocess Data Set - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
original = preprocessing()
df = original.drop(columns=['stroke'])

# Hierarchical Cluster Creation + Evaluation- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# HELPER: Normalize each score to [0,1] where 1 is always "better"
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 2. Use a composite internal score (no ground truth leakage)
def meta_score(dataset, labels):
    k = len(np.unique(labels))

    # Silhouette: higher is better, [-1, 1] → [0, 1]
    sil = silhouette_score(dataset, labels)
    sil_norm = (sil + 1) / 2

    # Davies-Bouldin: lower is better → invert
    db = davies_bouldin_score(dataset, labels)
    db_norm = 1 / (1 + db)

    # Drop CH entirely — it's monotonically biased toward k=2

    # Penalty: discourage trivially small k (below ~5 clusters)
    # Smoothly ramps up from 0 penalty at k=10+ to a real hit at k=2
    k_penalty = 1 - np.exp(-k / 5)  # approaches 1 as k grows, near 0 at k=2

    return (0.6 * sil_norm + 0.4 * db_norm) * k_penalty

def hierarchical(dataset, k_values):

    # Helper Functions (Each performs one kind of Hierarchical Clustering

    '''
    Purpose: minimizes the sum of squared differences within all clusters.
    '''
    def ward(dataset, current_k):
        model = AgglomerativeClustering(
            n_clusters=current_k,
            linkage='ward',
            metric='euclidean',
            connectivity=None
        )
        labels = model.fit_predict(dataset)
        return labels


    '''
    Purpose: minimizes the maximum distance between observations of pairs of clusters.
    '''
    def complete(dataset, current_k):
        model = AgglomerativeClustering(
            n_clusters=current_k,
            linkage='complete',
            metric='euclidean',
            connectivity=None
        )
        labels = model.fit_predict(dataset)
        return labels

    '''
    Purpose: minimizes the average of the distances between all observations of pairs of clusters.
    '''
    def average(dataset, current_k):
        model = AgglomerativeClustering(
            n_clusters=current_k,
            linkage='average',
            metric='euclidean',
            connectivity=None
        )
        labels = model.fit_predict(dataset)
        return labels

    '''
    Purpose: minimizes the distance between the closest observations of pairs of clusters.
    '''
    def single(dataset, current_k):
        model = AgglomerativeClustering(
            n_clusters=current_k,
            linkage='single',
            metric='euclidean',
            connectivity=None
        )
        labels = model.fit_predict(dataset)
        return labels


    methods = [ward, complete, average, single]
    names = ["ward", "complete", "average", "single"]
    optimal_clusters = []

    for method in methods:

        current_best_eval = 0
        best_k = 0
        best_labels = None
        scores = []

        for k in range(20, k_values + 1):
            # 1. perform method with k clusters on dataset
            current_cluster = method(dataset, k)

            scores.append((k, meta_score(dataset, current_cluster)))



            # 2. Calculate Evaluation Score

            # (i) -  Internal Indices - (Sum of Squared Errors (SSE), Silhouette Coefficient)
            y_true = original['stroke']
            y_pred = current_cluster
            mse = mean_squared_error(y_true, y_pred)
            silhouette = silhouette_score(dataset, current_cluster)

            # (iii) -  External Indices - (Homogeneity, Completeness, V-measure, Contingency Matrix)
            homogeneity = homogeneity_score(y_true, y_pred)
            completeness = completeness_score(y_true, y_pred)
            v_score = v_measure_score(y_true, y_pred)

        # 3. Report K with the best Eval Score (Create t-SNE graph)
            eval_score = meta_score(dataset, current_cluster)
            #mse, silhouette, homogeneity, completeness, v_score

            if eval_score > current_best_eval:
                current_best_eval = eval_score
                best_k = k
                best_labels = current_cluster

        ks, vals = zip(*scores)
        plt.plot(ks, vals)
        plt.title(f'{method.__name__} — meta score vs k')
        plt.xlabel('k')
        plt.ylabel('score')
        plt.show()

        optimal = (current_best_eval, best_k, best_labels)
        if method == methods[0] :
            print("--", names[0], "--")
            print("Optimal Number of Clusters: ", best_k)
        if method == methods[1] :
            print("\n--", names[1], "--")
            print("Optimal Number of Clusters: ", best_k)
        if method == methods[2] :
            print("\n--", names[2], "--")
            print("Optimal Number of Clusters: ", best_k)
        if method == methods[3] :
            print("\n--", names[3], "--")
            print("Optimal Number of Clusters: ", best_k)
        optimal_clusters.append(optimal)

    # Among 4 Methods: (ii) - Compare Relative Indices - (Sum of Squared Errors (SSE), Adjusted Rand score, Normalized mutual information score, Adjusted mutual information score)

    labels = {
        'ward': optimal_clusters[0][2],
        'complete': optimal_clusters[1][2],
        'average': optimal_clusters[2][2],
        'single': optimal_clusters[3][2]
    }

    pairs = [('ward', 'complete'), ('ward', 'average'), ('ward', 'single'),
             ('complete', 'average'), ('complete', 'single'), ('average', 'single')]

    print("\nRelative Clustering: ")
    for a, b in pairs:
        print(f"{a} vs {b}:")
        print(f"  ARI: {adjusted_rand_score(labels[a], labels[b]):.4f}")
        print(f"  NMI: {normalized_mutual_info_score(labels[a], labels[b]):.4f}")
        print(f"  AMI: {adjusted_mutual_info_score(labels[a], labels[b]):.4f}")

    return optimal_clusters

# Anomaly Detection - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Purpose: evaluate the distance between all data instances of a cluster, and evaluate the the degree to which they are outliers.
'''
def anomaly_detection(dataSet, z_score_threshold=2, percentage_difference=1.5):
    # Need at least 2 observations to compute a distance matrix
    if len(dataSet) < 2:
        print("Cluster Size: 1 — skipping (too small for anomaly detection)")
        return []

    # Create List of Dendrogram Linkage Values
    aDendrogram = linkage(dataSet, 'ward')
    numEntries = len(dataSet)
    merge_distance = np.zeros(numEntries)
    for i, (a, b, dist, _) in enumerate(aDendrogram):
        a, b = int(a), int(b)
        # If a or b is an original point (< n), record the distance
        if a < numEntries:
            merge_distance[a] = dist
        if b < numEntries:
            merge_distance[b] = dist

    # Calculate if the linkage value qualifies the instance as a outlie. "f(x)".
    average_distance = np.mean(merge_distance)
    mu, std = norm.fit(merge_distance)
    z_scores = (merge_distance - mu) / std
    anomaly_list = []

    # To qualify z_score must be above given threshold, and linkage value must be 50% Greater than the average value.
    for i, dist in enumerate(merge_distance):
        if z_scores[i] > z_score_threshold and (dist / average_distance) > percentage_difference:
            outlier = (dataSet.iloc[[i]], dist)  # use i not hardcoded 4
            anomaly_list.append(outlier)

    # Cluster Evaluation
    print("Cluster Size: ", numEntries)
    print("Number of anomalies: ", len(anomaly_list))
    if len(anomaly_list) > 0:
        print("Percentage of Outliers: ", len(anomaly_list) / numEntries * 100, "%")
    else:
        print("Percentage of Outliers: ", 0, "%")

    for method_name, optimal in zip(['ward', 'complete', 'average', 'single'], optimal_clusters):
        best_score, best_k, best_labels = optimal



    return anomaly_list

# Actually Run Simulation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

optimal_clusters = hierarchical(df, 40)

# Anomaly Detection
for method_name, optimal in zip(['ward', 'complete', 'average', 'single'], optimal_clusters):
    best_score, best_k, best_labels = optimal
    print(f"\n=== Anomaly Detection: {method_name} (k={best_k}) ===")
    for cluster_id in np.unique(best_labels):
        cluster_data = df[best_labels == cluster_id]
        print(f"\n--- Cluster {cluster_id} ---")
        anomaly_detection(cluster_data)

# t-SNE Plots (separate, after anomaly detection)
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(df)  # fit once, reuse for all 4 plots

for method_name, optimal in zip(['ward', 'complete', 'average', 'single'], optimal_clusters):
    best_score, best_k, best_labels = optimal
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=best_labels, cmap='tab10', s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f't-SNE: {method_name} (k={best_k})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()