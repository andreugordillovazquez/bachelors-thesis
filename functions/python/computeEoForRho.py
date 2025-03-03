import numpy as np

def computeEoForRho(rho, Q, pi_matrix, g_matrix):
    """
    Computes E0(ρ) for a given ρ value, used in the optimization of the error exponent 
    E(R) = max{E0(ρ) - ρR} for communication systems.
    
    Parameters:
        rho (float): Value of ρ parameter, 0 ≤ ρ ≤ 1.
        Q (np.ndarray): 1D array of probabilities for constellation points (length M).
        pi_matrix (np.ndarray): Matrix of quadrature weight products with shape (M, K),
                                where K is typically related to the number of nodes.
        g_matrix (np.ndarray): Matrix of G function values with shape (M, K).
    
    Returns:
        float: Computed value of E0(ρ) for the given parameters.
    """
    # Compute the intermediate terms.
    # Each element of g_matrix is raised to the power of 1/(1+rho)
    g_powered = g_matrix ** (1 / (1 + rho))
    
    # Weighted sum over g_powered. Here Q is a 1D array of length M and g_powered is (M, K)
    # so the dot product gives a 1D array of length K.
    qg_rho = np.dot(Q, g_powered)
    
    # Raise the weighted sum to the power of rho
    qg_rho_t = qg_rho ** rho
    
    # Compute the Pi-weighted terms.
    pi_g_rho = pi_matrix * (g_matrix ** (-rho / (1 + rho)))
    
    # Combine results:
    # First, perform matrix multiplication between pi_g_rho (M, K) and qg_rho_t (K,)
    # which results in a vector of length M; then take the dot product with Q (length M)
    # to obtain a scalar.
    component_second = np.dot(Q, np.dot(pi_g_rho, qg_rho_t))
    
    # Compute the final value for E0(rho) using base-2 logarithm.
    Eo_rho = -np.log2((1 / np.pi) * component_second)
    
    return Eo_rho