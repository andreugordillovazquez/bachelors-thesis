import numpy as np
from numpy.polynomial.hermite import hermgauss

def GaussHermite_Locations_Weights(N):
    """
    Computes Gauss-Hermite quadrature nodes and weights.
    
    Parameters:
        N (int): Number of points for quadrature
        
    Returns:
        tuple: (nodes, weights)
            - nodes: numpy array of quadrature points
            - weights: numpy array of corresponding weights
    """
    # hermgauss returns points and weights for the physicist's version of the Hermite polynomials
    # We need to scale them for the probabilist's version (for Gaussian density integration)
    nodes, weights = hermgauss(N)
    
    return nodes, weights 