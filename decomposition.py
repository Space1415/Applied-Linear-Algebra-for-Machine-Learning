import numpy as np

def compute_eigenvalues_and_vectors(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

def compute_svd(A):
    U, S, VT = np.linalg.svd(A)
    return U, S, VT
