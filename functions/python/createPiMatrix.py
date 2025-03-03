import numpy as np

def createPiMatrix(M, N, weights):
    """
    Creates a matrix of quadrature weight products.
    
    For each constellation symbol (row) and for all combinations of 
    Gauss-Hermite weights, the function computes the product and stores it 
    in the corresponding block of the output matrix.
    
    Parameters:
        M (int): Number of constellation symbols.
        N (int): Number of Gauss-Hermite nodes.
        weights (array-like): 1D array of weights of length N.
    
    Returns:
        np.ndarray: A matrix of shape (M, M * N**2) with weight products.
    """
    pi_matrix = np.zeros((M, M * (N**2)))
    
    for i in range(M):
        idx = 0
        for j in range(N):
            for k in range(N):
                pi_block = weights[j] * weights[k]
                col = i * (N**2) + idx
                pi_matrix[i, col] = pi_block
                idx += 1
                
    return pi_matrix