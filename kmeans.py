from sklearn.cluster import KMeans
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from main import preprocessing

"""
Steps:
- Run a bunch of k-means experiments with different numbers of clusters
    - also different preprocessing and max iterations
- Determine best k using SSE and plot SSE vs K
- Pick 5 top params structures
"""

def kmeans_clustering(
        n_clusters = 1, 
        *, 
        compute_anomaly_score = False, 
        max_iter = 300,
        pca = False,
        plot = True
    ):
    # use sklearn k_means clustering
    df = preprocessing(scaling=True, pca=pca)
    if pca:
        X = df[["pc1", "pc2"]]
    else:
        X = df.drop(columns=["stroke"])

    df_cp = df.copy()
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=1)
    cluster_labels = model.fit_predict(X)
    distances = model.transform(X)
    anomaly_scores = distances[np.arange(len(X)), cluster_labels]

    df_cp["cluster"] = cluster_labels

    if plot:
        # simple 2D visualization using first two features
        if X.shape[1] >= 2:
            plt.figure(figsize=(6, 5))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_labels)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title(f"K-means Clustering (k={n_clusters})")
            plt.show()

    if compute_anomaly_score:
        df["anomaly_score"] = anomaly_scores
        threshold = df["anomaly_score"].quantile(0.9973)
        df["anomaly"] = df["anomaly_score"] >= threshold

        print(type(df))
        for row in df.iterrows():
            print(row)
            break

        potential_outliers = ["hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        # for each anomaly in the dataset, print out its value in the above categories
        for index, row in df.iterrows():
            if row["anomaly"] == True:
                print(f"Anomaly {index}:")
                for o in potential_outliers:
                    print(f"{o}: {row[o]}", end=', ')
                print("\n")

        # plot data points according to anomaly function
        if X.shape[1] >= 2:
            plt.figure(figsize=(6, 5))
            scatter = plt.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                c=df["anomaly_score"],
                cmap="Reds",
                alpha=0.7,
            )
            plt.scatter(
                X.loc[df["anomaly"], X.columns[0]],
                X.loc[df["anomaly"], X.columns[1]],
                facecolors="none",
                edgecolors="black",
                s=100,
                linewidths=1.5,
                label="Anomalies",
            )
            plt.xlabel(X.columns[0])
            plt.ylabel(X.columns[1])
            plt.title(f"K-means Anomaly Scores (k={n_clusters})")
            plt.colorbar(scatter, label="Anomaly score")
            plt.legend()
            plt.show()

    SSE = model.inertia_

    return df_cp, SSE


def kmeans_eval():
    pass

def main():
    ks = [4]

    params_dict = {}
    for k in ks:
        df, sse = kmeans_clustering(
            n_clusters=k,
            max_iter=100,
            pca=True,
            plot=True,
            compute_anomaly_score=True,
        )
        params_dict[k] = sse
        print(f"K: {k}, SSE: {sse}")
        # print out number of instances per cluster as a percentage
        cluster_pct = df["cluster"].value_counts(normalize=True).sort_index() * 100
        print("Cluster percentages:")
        for cluster_id, pct in cluster_pct.items():
            print(f"  Cluster {cluster_id}: {pct:.2f}%")
    

    # plot SSE vs k values
    plt.figure(figsize=(8, 5))
    plt.plot(list(params_dict.keys()), list(params_dict.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("SSE")
    plt.title("SSE vs k for K-means (o = with PCA)")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.show()
    

if __name__ == "__main__":
    main()
