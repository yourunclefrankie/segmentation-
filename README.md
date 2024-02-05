# segmentation-# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import seaborn as sns

# Step 2: Load and Explore the Dataset
# Load your dataset (replace 'your_dataset.csv' with your actual file path)
dataset = pd.read_csv('your_dataset.csv')

# Explore the dataset
print(dataset.head())
print(dataset.info())

# Step 3: Handle Missing Values (if any)
# Check for missing values
print(dataset.isnull().sum())

# Handle missing values if needed
# Example: dataset = dataset.dropna()

# Step 4: Data Preprocessing
# Extract features (X) and target variable (y) if applicable
# Example: X = dataset.drop('target_column', axis=1)

# Step 5: Standardize the Data
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Perform PCA
# Initialize PCA
pca = PCA()

# Fit and transform the scaled data
PC_data = pca.fit_transform(X_scaled)

# Step 7: Explained Variance Ratio
# Check the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# Step 8: Scree Plot
# Plot the scree plot to visualize explained variance
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Step 9: Choose the Number of Components
# Select the number of principal components based on the scree plot and explained variance ratio.

# Step 10: Retrain PCA with Chosen Components
# Retrain PCA with the chosen number of components
pca = PCA(n_components=chosen_components)
PC_data = pca.fit_transform(X_scaled)

# Step 11: K-Means Clustering
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(PC_data)
print(y_kmeans)

# Step 12: Add k-means cluster labels to the original dataset
credit_card_kmeans = dataset.copy()
credit_card_kmeans["CLUSTERS_KMEANS"] = y_kmeans 
print(credit_card_kmeans.head(10))

# Step 13: Perform Hierarchical Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
y_agg_cluster = agg_cluster.fit_predict(PC_data)
print(y_agg_cluster)

# Step 14: Add hierarchical cluster labels to the original dataset
credit_card_agg = dataset.copy()
credit_card_agg["CLUSTERS_AGG"] = y_agg_cluster 
print(credit_card_agg.head(10))
