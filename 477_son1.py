import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("Steel_industry_data.csv")

# Only numerical clustering features
num_cols = [
    "Usage_kWh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Reactive_Power_kVarh",
    "NSM"
]

X = df[num_cols]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA explained variance:", pca.explained_variance_ratio_)

def hopkins_statistic(X, n_samples=0.1):
    """
    Calculate Hopkins statistic for a dataset X.

    Parameters:
    X : array-like, shape (n_samples, n_features)
    n_samples : float or int
        If float, it represents the fraction of data to sample.
        If int, it represents the number of samples.
    """

    if isinstance(n_samples, float):
        n_samples = int(n_samples * X.shape[0])

    # Randomly sample points from dataset
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[sample_indices]

    # Generate random points within feature ranges
    X_random = np.random.uniform(
        low=X.min(axis=0),
        high=X.max(axis=0),
        size=(n_samples, X.shape[1])
    )

    # Nearest neighbor model
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # Distances from real points to nearest neighbor
    u_distances, _ = nbrs.kneighbors(X_random, n_neighbors=1)

    # Distances from sampled points to nearest neighbor (excluding itself)
    w_distances, _ = nbrs.kneighbors(X_sample, n_neighbors=2)

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances[:, 1])

    hopkins_value = u_sum / (u_sum + w_sum)

    return hopkins_value
hopkins_value = hopkins_statistic(X_scaled)
print(f"Hopkins Statistic: {hopkins_value:.3f}")


loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=num_cols
)

print("\n=== PCA Loadings (Feature Contributions) ===")
print(loadings)


print("\n=== Top Contributing Features Per PCA Component ===")
for i in range(pca.n_components_):
    component_name = f"PC{i+1}"
    sorted_features = loadings[component_name].abs().sort_values(ascending=False)
    print(f"\n{component_name}:")
    print(sorted_features)


inertia_values = []      # SSE değerleri burada tutulacak
k_values = range(1, 11)  # 1'den 10 cluster'a kadar deneme

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)   # SSE (Within-Cluster Sum of Squares)



plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.xticks(k_values)
plt.grid(True)
plt.savefig("1_Elbow_Method.pdf", format="pdf", bbox_inches="tight")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

print("\nK-MEANS CLUSTER SIZES:")
print(pd.Series(k_labels).value_counts())

# Centroids (scaled → original scale)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=num_cols)
print("\nK-MEANS CENTROIDS:")
print(centroid_df)

# K-Means PCA plot
plt.figure(figsize=(7,5))
for c in np.unique(k_labels):
    plt.scatter(X_pca[k_labels==c,0], X_pca[k_labels==c,1], s=8, label=f"Cluster {c}")
plt.title("K-Means Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("2_KMeans_Clusters.pdf", format="pdf", bbox_inches="tight")
plt.show()


from sklearn.metrics import silhouette_score
labels_kmeans = kmeans.fit_predict(X_scaled)

# Silhouette Score
silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
print(f"K-Means Silhouette Score: {silhouette_kmeans:.3f}")


dbscan = DBSCAN(eps=0.25, min_samples=25)
db_labels = dbscan.fit_predict(X_scaled)

print("\nDBSCAN CLUSTER COUNTS:")
print(pd.Series(db_labels).value_counts())

noise_rate = (db_labels == -1).sum() / len(db_labels) * 100
print(f"DBSCAN Noise Percentage: {noise_rate:.2f}%")

# DBSCAN plot
plt.figure(figsize=(7,5))
unique = np.unique(db_labels)
for c in unique:
    if c == -1:
        plt.scatter(X_pca[db_labels==c,0], X_pca[db_labels==c,1], s=7, color="gray", label="Noise")
    else:
        plt.scatter(X_pca[db_labels==c,0], X_pca[db_labels==c,1], s=8, label=f"Cluster {c}")
plt.title("DBSCAN Clusters (PCA 2D)")
plt.legend()
plt.savefig("3_DBSCAN_Clusters.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Remove noise points (label = -1)
mask = db_labels != -1

X_dbscan_clean = X_scaled[mask]
labels_dbscan_clean = db_labels[mask]

# Check that at least 2 clusters remain
if len(np.unique(labels_dbscan_clean)) > 1:
    silhouette_dbscan = silhouette_score(X_dbscan_clean, labels_dbscan_clean)
    print(f"DBSCAN Silhouette Score (without noise): {silhouette_dbscan:.3f}")
else:
    print("DBSCAN Silhouette Score not defined (only one cluster detected).")


np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), size=500, replace=False)
X_sample = X_scaled[sample_indices]


plt.figure(figsize=(25,10))
sch.dendrogram(sch.linkage(X_sample, method='single'))
plt.title("Single Linkage Dendrogram")
plt.savefig("4_Dendrogram_Single.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(25,10))
sch.dendrogram(sch.linkage(X_sample, method='complete'))
plt.title("Complete Linkage Dendrogram")
plt.savefig("5_Dendrogram_Complete.pdf", format="pdf", bbox_inches="tight")
plt.show()


from sklearn.cluster import AgglomerativeClustering

hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)

print("Hierarchical cluster sizes:")
print(pd.Series(hier_labels).value_counts())


