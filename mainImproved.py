import time

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    mean_squared_error,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

# CS 4445 - Project 3
# Group: Shane G., Jake W., Victor C.
# April 2026

# =============================================================================
# DATA LOADING
# =============================================================================

file_path = "healthcare-dataset-stroke-data.csv"
dataframe = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/stroke-prediction-dataset",
    file_path,
)


# =============================================================================
# PREPROCESSING
# =============================================================================

def base_pp():
    """Preprocess the dataset into a numeric dataframe suitable for clustering."""
    df = dataframe.copy().dropna(subset=["bmi"])
    df = df[df["gender"] != "Other"].copy()
    df = df.drop(columns=["id"])
    df = df.rename(columns={"Residence_type": "residence_type"})

    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["ever_married"] = df["ever_married"].map({"No": 0, "Yes": 1})
    df["residence_type"] = df["residence_type"].map({"Rural": 0, "Urban": 1})

    df = df.join(
        pd.get_dummies(
            df[["work_type", "smoking_status"]],
            prefix=["work_type", "smoking_status"],
            dtype=int,
        )
    ).drop(columns=["work_type", "smoking_status"])

    return df


def preprocessing(*, scaling=True, pca=False, outlier_removal=False, feature_selection=None):
    if feature_selection is None:
        feature_selection = []

    df = base_pp()

    if scaling:
        scaler = StandardScaler()
        scale_cols = ["age", "avg_glucose_level", "bmi"]
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        if outlier_removal:
            z_score_threshold = 3
            outlier_mask = (df[scale_cols].abs() <= z_score_threshold).all(axis=1)
            df = df.loc[outlier_mask].copy()

    if pca:
        features = df.drop(columns=["stroke"])
        pca_model = PCA(n_components=2, random_state=1)
        reduced_features = pca_model.fit_transform(features)
        df["pc1"] = reduced_features[:, 0]
        df["pc2"] = reduced_features[:, 1]

    for feature in feature_selection:
        if feature in df.columns:
            df = df.drop(columns=[feature])
        else:
            print(f"Warning: '{feature}' is not a column and was skipped.")

    return df


def get_feature_matrix(df, *, pca=False):
    if pca:
        return df[["pc1", "pc2"]]
    return df.drop(columns=["stroke"])


# =============================================================================
# SHARED PLOTTING HELPERS
# =============================================================================

def print_cluster_percentages(labels):
    cluster_pct = pd.Series(labels).value_counts(normalize=True).sort_index() * 100
    print("Cluster percentages:")
    for cluster_id, pct in cluster_pct.items():
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"  {label}: {pct:.2f}%")


def plot_cluster_projection(X, labels, *, title, highlight_mask=None, color_values=None, cmap="tab10"):
    if X.shape[1] < 2:
        return

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels if color_values is None else color_values,
        cmap=cmap,
        alpha=0.75,
        s=18,
    )
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title(title)

    if highlight_mask is not None and np.any(highlight_mask):
        plt.scatter(
            X.loc[highlight_mask, X.columns[0]],
            X.loc[highlight_mask, X.columns[1]],
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            s=70,
            label="Highlighted points",
        )
        plt.legend()

    if color_values is not None:
        plt.colorbar(scatter, label="Score")

    plt.tight_layout()
    plt.show()


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def kmeans_clustering(
    n_clusters=4,
    *,
    compute_anomaly_score=False,
    max_iter=300,
    pca=False,
    plot=True,
):
    df = preprocessing(scaling=True, pca=pca)
    X = get_feature_matrix(df, pca=pca)

    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=1)
    cluster_labels = model.fit_predict(X)
    distances = model.transform(X)
    anomaly_scores = distances[np.arange(len(X)), cluster_labels]

    result = df.copy()
    result["cluster"] = cluster_labels

    if compute_anomaly_score:
        result["anomaly_score"] = anomaly_scores
        threshold = result["anomaly_score"].quantile(0.9973)
        result["anomaly"] = result["anomaly_score"] >= threshold

    if plot:
        plot_cluster_projection(
            X,
            cluster_labels,
            title=f"K-means Clustering (k={n_clusters})",
        )
        if compute_anomaly_score:
            plot_cluster_projection(
                X,
                cluster_labels,
                title=f"K-means Anomaly Scores (k={n_clusters})",
                highlight_mask=result["anomaly"],
                color_values=result["anomaly_score"],
                cmap="Reds",
            )
            # Print anomaly details for key features
            potential_outliers = ["hypertension", "heart_disease", "avg_glucose_level", "bmi"]
            print("\nAnomalous data points:")
            for index, row in result.iterrows():
                if row["anomaly"]:
                    print(f"  Index {index}: " + ", ".join(
                        f"{f}={row[f]:.3f}" if isinstance(row[f], float) else f"{f}={row[f]}"
                        for f in potential_outliers if f in row
                    ))

    return {
        "data": result,
        "sse": model.inertia_,
        "n_clusters": n_clusters,
        "n_anomalies": int(result["anomaly"].sum()) if compute_anomaly_score else 0,
    }


def kmeans_eval(k_values=None, pca=True):
    """
    Sweep over k values and plot the SSE elbow curve to identify the best k.
    Restored from kmeans.py — this was missing from the previous main.py.
    """
    if k_values is None:
        k_values = list(range(2, 11))

    print("\n--- K-means Parameter Sweep ---")
    sse_dict = {}
    for k in k_values:
        result = kmeans_clustering(n_clusters=k, pca=pca, plot=False)
        sse_dict[k] = result["sse"]
        print(f"  k={k}: SSE={result['sse']:.4f}")

    # Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(sse_dict.keys()), list(sse_dict.values()), marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE (Inertia)")
    plt.title("K-means Elbow Curve" + (" (with PCA)" if pca else ""))
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return sse_dict


def summarize_kmeans(result):
    print("\n=== K-means ===")
    print(f"Clusters: {result['n_clusters']}")
    print(f"SSE: {result['sse']:.4f}")
    print(f"Anomalies flagged: {result['n_anomalies']}")
    print_cluster_percentages(result["data"]["cluster"])


# =============================================================================
# DBSCAN CLUSTERING
# =============================================================================

def dbscan_clustering(
    eps=0.5,
    min_samples=5,
    *,
    compute_anomaly=False,
    pca=False,
    plot=True,
):
    start_time = time.time()
    df = preprocessing(scaling=True, pca=pca)
    X = get_feature_matrix(df, pca=pca)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(X)

    result = df.copy()
    result["cluster"] = cluster_labels
    result["anomaly"] = result["cluster"] == -1 if compute_anomaly else False

    unique_clusters = set(cluster_labels)
    n_clusters = len(unique_clusters - {-1})
    n_noise = int((result["cluster"] == -1).sum())

    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples})")
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
    print_cluster_percentages(result["cluster"])

    if plot:
        plot_cluster_projection(
            X,
            cluster_labels,
            title=f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})",
        )
        if compute_anomaly:
            plot_cluster_projection(
                X,
                cluster_labels,
                title="DBSCAN Anomaly View",
                highlight_mask=result["anomaly"],
                color_values=result["anomaly"].astype(int),
                cmap="coolwarm",
            )

    return {
        "data": result,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "runtime_seconds": time.time() - start_time,
    }


def dbscan_eval(pca=True):
    """
    Sweep over eps and min_samples values, report cluster/noise counts,
    and plot parameter sensitivity. Restored from dbscan.py — this was
    missing from the previous main.py.
    """
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = [3, 5, 10]

    print("\n--- DBSCAN Parameter Sweep ---")
    params_dict = {}
    for min_s in min_samples_values:
        for eps in eps_values:
            result = dbscan_clustering(eps=eps, min_samples=min_s, pca=pca, plot=False)
            params_dict[(eps, min_s)] = (result["n_clusters"], result["n_noise"])

    # Sensitivity plot for min_samples=5
    clusters_by_eps = []
    noise_by_eps = []
    for eps in eps_values:
        c, n = params_dict[(eps, 5)]
        clusters_by_eps.append(c)
        noise_by_eps.append(n)

    plt.figure(figsize=(8, 5))
    plt.plot(eps_values, clusters_by_eps, marker="o", label="Clusters")
    plt.plot(eps_values, noise_by_eps, marker="x", label="Noise Points")
    plt.xlabel("eps")
    plt.ylabel("Count")
    plt.title("DBSCAN Parameter Sensitivity (min_samples=5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return params_dict


def summarize_dbscan(result, *, eps, min_samples):
    print("\n=== DBSCAN ===")
    print(f"eps: {eps}, min_samples: {min_samples}")
    print(f"Clusters: {result['n_clusters']}")
    print(f"Noise points: {result['n_noise']}")
    print(f"Runtime: {result['runtime_seconds']:.4f}s")
    print_cluster_percentages(result["data"]["cluster"])


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def hierarchical_meta_score(dataset, labels):
    """
    Composite internal score (no ground truth leakage).
    Silhouette: higher = better [-1,1] → normalized to [0,1]
    Davies-Bouldin: lower = better → inverted
    k_penalty: discourages trivially small k (approaches 1 as k grows)
    """
    k = len(np.unique(labels))
    if k < 2:
        return float("-inf")

    sil = silhouette_score(dataset, labels)
    sil_norm = (sil + 1) / 2

    db = davies_bouldin_score(dataset, labels)
    db_norm = 1 / (1 + db)

    k_penalty = 1 - np.exp(-k / 5)
    return (0.6 * sil_norm + 0.4 * db_norm) * k_penalty


def fit_hierarchical_method(dataset, *, linkage_name, current_k):
    model = AgglomerativeClustering(
        n_clusters=current_k,
        linkage=linkage_name,
        metric="euclidean",
    )
    return model.fit_predict(dataset)


def hierarchical_clustering(
    *,
    pca=False,
    plot=True,
    k_start=20,
    k_end=40,
):
    """
    Runs Ward, Complete, Average, and Single linkage hierarchical clustering
    over a range of k values. Reports internal and external metrics per method,
    computes relative agreement between methods, runs anomaly detection on the
    best model, and plots t-SNE projections.

    k_start/k_end default to 20–40 as in the original Hierarchical.py.
    """
    df = preprocessing(scaling=True, pca=pca)
    X = get_feature_matrix(df, pca=pca)
    y_true = df["stroke"]

    method_names = ["ward", "complete", "average", "single"]
    method_results = {}

    print("\n--- Hierarchical Clustering ---")
    for method_name in method_names:
        best_score = float("-inf")
        best_k = None
        best_labels = None
        scores = []

        for k in range(k_start, k_end + 1):
            labels = fit_hierarchical_method(X, linkage_name=method_name, current_k=k)
            score = hierarchical_meta_score(X, labels)
            scores.append((k, score))

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        # Internal metrics for best k
        sil = silhouette_score(X, best_labels)
        db = davies_bouldin_score(X, best_labels)

        # External metrics against stroke ground truth
        mse = mean_squared_error(y_true, best_labels)
        hom = homogeneity_score(y_true, best_labels)
        com = completeness_score(y_true, best_labels)
        v = v_measure_score(y_true, best_labels)

        print(f"\n-- {method_name.upper()} --")
        print(f"  Optimal k: {best_k}  |  Meta Score: {best_score:.4f}")
        print(f"  Internal  → Silhouette: {sil:.4f}, Davies-Bouldin: {db:.4f}")
        print(f"  External  → MSE: {mse:.4f}, Homogeneity: {hom:.4f}, "
              f"Completeness: {com:.4f}, V-measure: {v:.4f}")

        method_results[method_name] = {
            "best_score": best_score,
            "best_k": best_k,
            "labels": best_labels,
            "scores": scores,
        }

        if plot:
            ks, vals = zip(*scores)
            plt.figure(figsize=(7, 4))
            plt.plot(ks, vals, marker="o")
            plt.title(f"{method_name.capitalize()} Linkage — Meta Score vs k")
            plt.xlabel("k")
            plt.ylabel("Meta Score")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # Relative agreement between all method pairs
    pairs = [
        ("ward", "complete"),
        ("ward", "average"),
        ("ward", "single"),
        ("complete", "average"),
        ("complete", "single"),
        ("average", "single"),
    ]
    pair_scores = []
    print("\nRelative Clustering Agreement:")
    for left, right in pairs:
        ll = method_results[left]["labels"]
        rl = method_results[right]["labels"]
        ari = adjusted_rand_score(ll, rl)
        nmi = normalized_mutual_info_score(ll, rl)
        ami = adjusted_mutual_info_score(ll, rl)
        pair_scores.append({"pair": f"{left} vs {right}", "ari": ari, "nmi": nmi, "ami": ami})
        print(f"  {left} vs {right}: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}")

    # Best overall method
    best_method_name = max(
        method_results,
        key=lambda m: method_results[m]["best_score"],
    )
    best_method = method_results[best_method_name]
    print(f"\nBest overall method: {best_method_name.upper()} (k={best_method['best_k']})")

    # Anomaly detection on best method's clusters
    best_labels = best_method["labels"]
    anomalies_by_cluster = {}
    print(f"\nAnomaly Detection ({best_method_name}, k={best_method['best_k']}):")
    for cluster_id in np.unique(best_labels):
        cluster_data = X.loc[best_labels == cluster_id]
        anomaly_indices = anomaly_detection(cluster_data)
        anomalies_by_cluster[int(cluster_id)] = anomaly_indices
        pct = len(anomaly_indices) / len(cluster_data) * 100 if len(cluster_data) > 0 else 0
        print(f"  Cluster {cluster_id}: size={len(cluster_data)}, "
              f"anomalies={len(anomaly_indices)} ({pct:.2f}%)")

    # t-SNE plots for all four methods
    if plot:
        print("\nFitting t-SNE (this may take a moment)...")
        tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(X)
        projection = pd.DataFrame(tsne_coords, columns=["tsne1", "tsne2"])

        for method_name, method_result in method_results.items():
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                projection["tsne1"],
                projection["tsne2"],
                c=method_result["labels"],
                cmap="tab10",
                s=10,
            )
            plt.colorbar(scatter, label="Cluster")
            plt.title(f"t-SNE: {method_name.capitalize()} (k={method_result['best_k']})")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.tight_layout()
            plt.show()

    return {
        "data": df,
        "features": X,
        "methods": method_results,
        "agreement": pair_scores,
        "best_method_name": best_method_name,
        "anomalies_by_cluster": anomalies_by_cluster,
    }


def anomaly_detection(cluster_df, z_score_threshold=2, percentage_difference=1.5):
    """
    Evaluates the distance between all data instances in a cluster and flags
    those that are statistical outliers based on their dendrogram merge distance.
    Returns a list of anomalous row indices.
    """
    if len(cluster_df) < 2:
        return []

    dendrogram_linkage = linkage(cluster_df, "ward")
    num_entries = len(cluster_df)
    merge_distance = np.zeros(num_entries)

    for a, b, dist, _ in dendrogram_linkage:
        a = int(a)
        b = int(b)
        if a < num_entries:
            merge_distance[a] = dist
        if b < num_entries:
            merge_distance[b] = dist

    average_distance = np.mean(merge_distance)
    _, std = norm.fit(merge_distance)
    if std == 0:
        return []

    z_scores = (merge_distance - np.mean(merge_distance)) / std
    anomaly_indices = []

    for i, dist in enumerate(merge_distance):
        if z_scores[i] > z_score_threshold and (dist / average_distance) > percentage_difference:
            anomaly_indices.append(cluster_df.index[i])

    return anomaly_indices


def summarize_hierarchical(result):
    print("\n=== Hierarchical Clustering Summary ===")
    for method_name, method_result in result["methods"].items():
        print(
            f"  {method_name.capitalize()}: "
            f"best_k={method_result['best_k']}, "
            f"best_score={method_result['best_score']:.4f}"
        )

    print(f"\nBest overall method: {result['best_method_name']}")

    print("\nMethod agreement:")
    for pair_result in result["agreement"]:
        print(
            f"  {pair_result['pair']}: "
            f"ARI={pair_result['ari']:.4f}, "
            f"NMI={pair_result['nmi']:.4f}, "
            f"AMI={pair_result['ami']:.4f}"
        )

    print("\nAnomalies by cluster (best method):")
    for cluster_id, anomaly_indices in result["anomalies_by_cluster"].items():
        print(f"  Cluster {cluster_id}: {len(anomaly_indices)} anomalies")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Stroke Dataset Clustering — CS 4445 Project 3")
    print("=" * 60)

    base_df = preprocessing(scaling=True, pca=False)
    print(f"\nDataset: {len(base_df)} rows, {base_df.shape[1] - 1} features")

    # ------------------------------------------------------------------
    # K-MEANS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)

    # Elbow curve sweep to find best k
    kmeans_eval(k_values=list(range(2, 11)), pca=True)

    # Final model with chosen k
    kmeans_result = kmeans_clustering(
        n_clusters=4,
        max_iter=100,
        pca=True,
        plot=True,
        compute_anomaly_score=True,
    )
    summarize_kmeans(kmeans_result)

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING")
    print("=" * 60)

    # Parameter sensitivity sweep
    dbscan_eval(pca=True)

    # Final model with chosen params
    dbscan_eps = 0.5
    dbscan_min_samples = 5
    dbscan_result = dbscan_clustering(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        pca=True,
        plot=True,
        compute_anomaly=True,
    )
    summarize_dbscan(dbscan_result, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # ------------------------------------------------------------------
    # HIERARCHICAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 60)

    hierarchical_result = hierarchical_clustering(
        pca=False,
        plot=True,
        k_start=20,   # restored to original range from Hierarchical.py
        k_end=40,
    )
    summarize_hierarchical(hierarchical_result)


if __name__ == "__main__":
    main()
