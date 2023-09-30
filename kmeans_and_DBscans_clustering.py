#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:39:41 2023

@author: miguelangelsuevispacheco
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load Olivetti faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)

# Access the images and labels
images = faces_data.images
labels = faces_data.target

# Print some information about the dataset
print(f"Number of images: {len(images)}")
print(f"Number of individuals: {len(set(labels))}")



# Split into training and temporary (validation + test) sets
images_train_temp, images_test, labels_train_temp, labels_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Further split temporary set into validation and test sets
images_train, images_val, labels_train, labels_val = train_test_split(
    images_train_temp, labels_train_temp, test_size=0.25, stratify=labels_train_temp, random_state=42
)

# Create a support vector machine (SVM) classifier
classifier = SVC(kernel='linear', C=1)

# Define the number of folds for cross-validation
num_folds = 5  # You can adjust this based on your preferences

# Create a stratified k-fold object
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform k-fold cross-validation and get accuracy scores
cross_val_scores = cross_val_score(classifier, images_train.reshape(len(images_train), -1), labels_train, cv=stratified_kfold)

# Print the cross-validation scores
print("Cross-validation scores:", cross_val_scores)
print("Mean accuracy:", cross_val_scores.mean())

# Reshape images for compatibility with K-Means
flattened_images = images_train.reshape(len(images_train), -1)

# Specify the number of clusters (you need to choose this)
num_clusters = 50
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit K-Means and get cluster assignments
cluster_assignments = kmeans.fit_predict(flattened_images)


# Try different numbers of clusters and compute silhouette scores
silhouette_scores = []
for num_clusters in range(2, 21):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(flattened_images)
    silhouette_scores.append(silhouette_score(flattened_images, cluster_assignments))

# Plot silhouette scores for different numbers of clusters
plt.plot(range(2, 21), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.show()


# Use the cluster assignments as features
features_kmeans = cluster_assignments.reshape(-1, 1)

# Split the data into training and validation sets
features_train, features_val, labels_train_kmeans, labels_val_kmeans = train_test_split(
    features_kmeans, labels_train, test_size=0.2, random_state=42
)

# Create a classifier (SVM as an example)
classifier_kmeans = SVC(kernel='linear', C=1)

# Train the classifier on the training set
classifier_kmeans.fit(features_train, labels_train_kmeans)

# Make predictions on the validation set
predictions_val_kmeans = classifier_kmeans.predict(features_val)

# Evaluate the performance
accuracy_kmeans = accuracy_score(labels_val_kmeans, predictions_val_kmeans)
print("Accuracy with K-Means features:", accuracy_kmeans)

# Reshape images for compatibility with DBSCAN
flattened_images = images_train.reshape(len(images_train), -1)

# Apply PCA for dimensionality reduction (optional but can be useful)
pca = PCA(n_components=50)
reduced_images = pca.fit_transform(flattened_images)

# Apply DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)  # You may need to tune these parameters
cluster_assignments_dbscan = dbscan.fit_predict(reduced_images)

# Visualize the clustering (using the first two principal components for simplicity)
plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=cluster_assignments_dbscan, cmap='viridis', s=20)
plt.title('DBSCAN Clustering')
plt.show()
