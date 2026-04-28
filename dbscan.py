from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import time
from main import preprocessing

"""
Steps:
- Run DBSCAN with different eps and min_samples
- Evaluate clustering based on number of clusters and noise points
- Use noise points as anomaly detection
"""


def dbscan_clustering(
        eps=0.5,
        min_samples=5,
        *,
        compute_anomaly=False,
        pca=False,
        plot=True
    ):

    # preprocess data
    start_time = time.time()
    df = preprocessing(scaling=True, pca=pca)

    if pca:
        X = df[["pc1", "pc2"]]
    else:
        X = df.drop(columns=["stroke"])

    df_cp = df.copy()

    # run DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(X)

    df_cp["cluster"] = cluster_labels

    # compute cluster stats
    unique_clusters = set(cluster_labels)
    n_clusters = len(unique_clusters - {-1})
    n_noise = list(cluster_labels).count(-1)

    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples})")
    print(f"Clusters: {n_clusters}, Noise: {n_noise}")

    # percentage per cluster
    cluster_pct = df_cp["cluster"].value_counts(normalize=True).sort_index() * 100
    print("Cluster percentages:")
    for cluster_id, pct in cluster_pct.items():
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"  {label}: {pct:.2f}%")

    # plot clusters
    if plot and X.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_labels)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"DBSCAN Clustering (eps={eps})")
        plt.show()

    # anomaly detection (noise points)
    if compute_anomaly:
        df_cp["anomaly"] = df_cp["cluster"] == -1

        if plot and X.shape[1] >= 2:
            plt.figure(figsize=(6, 5))
            plt.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                c=df_cp["anomaly"],
                cmap="coolwarm"
            )
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("DBSCAN Anomalies")
            plt.show()

    print(f"Time: {time.time() - start_time:.4f}s")

    return df_cp, n_clusters, n_noise


def dbscan_eval():
    # vary BOTH eps and min_samples (THIS is the only important addition)
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = [3, 5, 10]

    params_dict = {}

    for min_s in min_samples_values:
        for eps in eps_values:
            _, n_clusters, n_noise = dbscan_clustering(
                eps=eps,
                min_samples=min_s,
                pca=True,
                plot=False
            )
            params_dict[(eps, min_s)] = (n_clusters, n_noise)
            print(f"eps: {eps}, min_samples: {min_s}, clusters: {n_clusters}, noise: {n_noise}")

    # keep simple plot (fix min_samples=5 like kmeans style)
    clusters = []
    noise = []

    for eps in eps_values:
        _, c, n = dbscan_clustering(
            eps=eps,
            min_samples=5,
            pca=True,
            plot=False
        )
        clusters.append(c)
        noise.append(n)

    plt.figure(figsize=(8, 5))
    plt.plot(eps_values, clusters, marker="o", label="Clusters")
    plt.plot(eps_values, noise, marker="x", label="Noise")
    plt.xlabel("eps")
    plt.ylabel("Count")
    plt.title("DBSCAN Parameter Sensitivity (min_samples=5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    dbscan_clustering(
        eps=0.5,
        min_samples=5,
        pca=True,
        plot=True,
        compute_anomaly=True
    )

    dbscan_eval()


if __name__ == "__main__":
    main()