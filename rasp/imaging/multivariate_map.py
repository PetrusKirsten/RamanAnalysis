"""K‑means and PCA for Raman maps (flattened into pixels × variables)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def compute_pca(image, n_components=3):
    pixel_matrix = image.spectral_data.reshape(-1, image.spectral_data.shape[-1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(pixel_matrix)
    return scores.reshape(image.spectral_data.shape[:-1] + (n_components,)), pca

def compute_kmeans(image, n_clusters=4, random_state=0):
    pixel_matrix = image.spectral_data.reshape(-1, image.spectral_data.shape[-1])
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(pixel_matrix)
    return labels.reshape(image.spectral_data.shape[:-1]), km
