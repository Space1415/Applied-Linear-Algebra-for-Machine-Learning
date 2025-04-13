import numpy as np

def add_matrices(A, B):
    return np.add(A, B)

def multiply_matrices(A, B):
    return np.dot(A, B)

def transpose_matrix(A):
    return np.transpose(A)

def inverse_matrix(A):
    return np.linalg.inv(A)

def identity_matrix(n):
    return np.identity(n)

def determinant(A):
    return np.linalg.det(A)
