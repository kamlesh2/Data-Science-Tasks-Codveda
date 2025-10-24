import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("Netflix_movies_and_tv_shows_clustering.csv")
df = df.dropna(subset=['release_year', 'duration', 'listed_in'])

df['duration'] = df['duration'].str.extract(r'(\d+)').astype(float)

genre_dummies = df['listed_in'].str.get_dummies(sep=', ')
df = pd.concat([df, genre_dummies], axis=1)

X = df[['release_year', 'duration'] + list(genre_dummies.columns)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(K_range, sil_scores, marker='s', color='green')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title("Netflix Titles Clusters (PCA Visualization)")
plt.tight_layout()
plt.show()

print("\nCluster Summary:")
print(df.groupby('Cluster')[['release_year', 'duration']].mean().round(2))

print("\nTop Genres per Cluster:")
genre_columns = genre_dummies.columns
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\nCluster {cluster_id}:")
    cluster_genres = df[df['Cluster'] == cluster_id][genre_columns].sum().sort_values(ascending=False)
    print(cluster_genres.head(5))
