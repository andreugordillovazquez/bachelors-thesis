import numpy as np

def createGMatrix(X, z_matrix, SNR_dB, G):
    """
    Computes the g_matrix used in further processing.
    
    For each received symbol index j and each source symbol index i,
    computes g_vals = G(z_matrix + sqrt(SNR)*(X[i] - X[j])),
    vectorizes g_vals in column-major order, and stores it in a block
    of g_matrix.
    
    Parameters:
        X (array-like): 1D array of source symbols (length M).
        z_matrix (np.ndarray): 2D array (N x N) of complex nodes.
        SNR_dB (float): Signal-to-noise ratio in dB.
        G (function): Function that accepts a NumPy array and returns an array
                      of the same shape (applied elementwise).
    
    Returns:
        np.ndarray: g_matrix of shape (M, M * N**2).
    """
    # Convert SNR from dB to linear scale
    SNR = 10**(SNR_dB/10)
    
    X = np.asarray(X)
    M = len(X)
    N = z_matrix.shape[0]
    g_matrix = np.zeros((M, M * (N**2)), dtype=complex)

    print("SNR", SNR)
    
    # Loop over received symbols (j)
    for j in range(M):
        print(X)
        print(z_matrix)

        # Precompute the term: sqrt(SNR)*(X - X[j])
        d = np.sqrt(SNR) * (X - X[j])
        # Determine column indices for the current block corresponding to j
        col_idx = slice(j * (N**2), (j + 1) * (N**2))
        
        # Loop over source symbols (i)
        for i in range(M):
            # Compute the G function over all quadrature nodes:
            # z_matrix is N-by-N and d[i] is a scalar (added to every element)
            g_vals = G(z_matrix + d[i])
            # Flatten in column-major order (equivalent to MATLAB's g_vals(:).')
            g_matrix[i, col_idx] = g_vals.flatten(order='F')
    
    return g_matrix