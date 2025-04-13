import numpy as np

def mean_center_data(X):
    return X - np.mean(X, axis=0)

def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)

def compute_pca(X, n_components=2):
    X_centered = mean_center_data(X)
    covariance_matrix = compute_covariance_matrix(X_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, sorted_indices[:n_components]]
    return np.dot(X_centered, top_components)
