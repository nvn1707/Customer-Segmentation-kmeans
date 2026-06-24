import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create sample customer dataset
data = {
    'CustomerID': range(1, 101),
    'Age': np.random.randint(18, 70, 100),
    'Annual_Income': np.random.randint(15, 120, 100),
    'Spending_Score': np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)
df.to_csv('customers.csv', index=False)
print("Dataset created!")
print(df.head())
print(f"Shape: {df.shape}")

# Prepare data for clustering
X = df[['Age', 'Annual_Income', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find best number of clusters
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', color='orange')
plt.title('Elbow Method - Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_curve.png')
plt.show()
print("Elbow curve saved!")

# Apply KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot customer segments
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        c=colors[i],
        label=f'Cluster {i+1}',
        s=100
    )

plt.title('Customer Segments - K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.savefig('customer_segments.png')
plt.show()
print("Customer segments plot saved!")

# Print cluster summary
print("\nCluster Summary:")
print(df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean().round(2))
print("\nDone! Check the output images.")