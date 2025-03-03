import numpy as np

def computeE0(params):
    """
    Computes E0(ρ) for a range of ρ values using Gaussian-Hermite quadrature.
    
    Parameters:
        params (dict): Dictionary containing:
            'num_points' (int): Number of points for ρ evaluation.
            'nodes' (array-like): Gaussian-Hermite quadrature nodes.
            'weights' (array-like): Gaussian-Hermite quadrature weights.
            'X' (array-like): Normalized constellation points.
            'SNR' (float): Signal-to-Noise Ratio.
            'Q' (array-like): Probability distribution for constellation points.
    
    Returns:
        tuple: (rho_values, Eo_rho)
            - rho_values (np.ndarray): ρ values evenly spaced from 0 to 1.
            - Eo_rho (np.ndarray): Computed E0 values corresponding to each ρ.
    """
    # Generate ρ values from 0 to 1
    rho_values = np.linspace(0, 1, params['num_points'])
    Eo_rho = np.zeros_like(rho_values)
    
    # Create necessary matrices for computation
    z_matrix = createComplexNodesMatrix(params['nodes'])
    pi_matrix = createPiMatrix(len(params['X']), len(params['nodes']), params['weights'])
    
    # Define the Gaussian PDF function G(z)
    def G(z):
        return (1 / np.pi) * np.exp(-np.abs(z)**2)
    
    # Create G matrix for all constellation points
    g_matrix = createGMatrix(params['X'], z_matrix, params['SNR'], G)
    
    # Compute E0(ρ) for each ρ value
    for idx, rho in enumerate(rho_values):
        Eo_rho[idx] = computeEoForRho(rho, params['Q'], pi_matrix, g_matrix)
    
    return rho_values, Eo_rho
