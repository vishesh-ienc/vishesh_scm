# Hierarchical Clustering - Supermarket Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ---------------------------------------------------
# 1. Create sample Supermarket Customer data
# ---------------------------------------------------

data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15],
    'Age': [19, 21, 20, 23, 31, 35, 40, 42, 36, 38,
            25, 27, 24, 29, 45],
    'Annual_Income': [15, 16, 17, 18, 25, 28, 35, 40, 30, 32,
                      22, 24, 20, 26, 45],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 6, 20, 78, 17,
                       65, 73, 45, 52, 12]
}

df = pd.DataFrame(data)

print("Supermarket Customer Data:")
print(df)

# ---------------------------------------------------
# 2. Select features for clustering
# ---------------------------------------------------

X = df[['Annual_Income', 'Spending_Score']]

# ---------------------------------------------------
# 3. Standardize the data
# ---------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# 4. Perform Hierarchical Clustering
# ---------------------------------------------------

# Ward linkage minimizes variance within clusters
linked = linkage(X_scaled, method='ward')

# ---------------------------------------------------
# 5. Display Dendrogram
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    labels=df['CustomerID'].values,
    leaf_rotation=90,
    leaf_font_size=10
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer ID")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 6. Create clusters
# ---------------------------------------------------

# Here we choose 3 clusters
clusters = fcluster(linked, t=3, criterion='maxclust')

df['Cluster'] = clusters

print("\nCustomer Clusters:")
print(df[['CustomerID', 'Annual_Income',
          'Spending_Score', 'Cluster']])

# ---------------------------------------------------
# 7. Visualize the clusters
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

colors = ['red', 'blue', 'green']

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        color=colors[cluster - 1],
        label=f'Cluster {cluster}',
        s=100
    )

plt.title("Customer Segmentation using Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.show()