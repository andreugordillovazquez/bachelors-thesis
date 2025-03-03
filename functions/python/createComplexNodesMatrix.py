import numpy as np

def createComplexNodesMatrix(nodes):
    """
    Creates a complex nodes matrix given an array of nodes.
    
    Each element (i, j) of the resulting matrix is computed as:
        nodes[i] + 1j * nodes[j]
    
    Parameters:
        nodes (array-like): 1D array of nodes.
        
    Returns:
        np.ndarray: A square matrix of shape (N, N) with complex entries.
    """
    nodes = np.asarray(nodes)
    z_matrix = nodes[:, np.newaxis] + 1j * nodes[np.newaxis, :]
    return z_matrix