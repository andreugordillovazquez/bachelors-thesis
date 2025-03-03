import numpy as np

def computeEoForRhoExponential(rho, Q, pi_matrix, g_matrix):
    """
    Optimized computation of E0(ρ) for the error exponent E(R).

    Parameters:
        rho (float): Value of ρ parameter, with 0 ≤ ρ ≤ 1.
        Q (np.ndarray): Probability distribution of constellation points, shape (M,).
        pi_matrix (np.ndarray): Matrix of quadrature weights products, shape (M, K).
        g_matrix (np.ndarray): Matrix of G function values, shape (M, K).

    Returns:
        float: Computed value of E0(ρ).
    """
    # Precomputed term
    s = 1 / (1 + rho)
    
    # Compute log_W for numerical stability (elementwise logarithm)
    log_W = np.log(g_matrix)
    
    # Compute π .* W^{-sρ} using elementwise multiplication
    pi_term = pi_matrix * np.exp(-s * rho * log_W)
    
    # Compute inner_exp = exp(s * log_W) (equivalent to g_matrix^s)
    inner_exp = np.exp(s * log_W)
    
    # Compute Q_inner = Q' * inner_exp.
    # With Q as a 1D array (shape M,) and inner_exp of shape (M, K),
    # the dot product produces a 1D array of shape (K,).
    Q_inner = np.dot(Q, inner_exp)
    
    # Compute (Q' * W^s)^ρ in a logarithmic manner for stability.
    log_Q_inner = np.log(Q_inner)
    outer_exp = np.exp(rho * log_Q_inner)  # This is (Q_inner)^ρ, shape (K,)
    
    # Final computation: F0 = Q' * pi_term * outer_term.
    # Here, np.dot(pi_term, outer_exp) yields a vector of shape (M,)
    # and then np.dot(Q, ...) gives the resulting scalar.
    F0 = np.dot(Q, np.dot(pi_term, outer_exp))
    
    # Compute E0(ρ) using base-2 logarithm.
    E0_rho = -np.log2((1 / np.pi) * F0)
    
    return E0_rho