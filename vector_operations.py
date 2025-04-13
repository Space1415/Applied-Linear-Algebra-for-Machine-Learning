import numpy as np

def add_vectors(v1, v2):
    return np.add(v1, v2)

def subtract_vectors(v1, v2):
    return np.subtract(v1, v2)

def scalar_multiply(vector, scalar):
    return scalar * vector

def dot_product(v1, v2):
    return np.dot(v1, v2)

def magnitude(vector):
    return np.linalg.norm(vector)

def normalize(vector):
    return vector / magnitude(vector)
