# knn_location.py
from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_knn(X, photo_ids, roads, target_road, k=5):
    idx = np.where(roads == target_road)[0]
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X[idx])
    return nn, idx
