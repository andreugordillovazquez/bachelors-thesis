import numpy as np

def computeFirstDerivativeE0(Q, pi_matrix, w_matrix, rho):
    """
    Computes the first derivative of E0(ρ) for the error exponent E(R)
    using a logarithmic approach for numerical stability.
    
    Parameters:
        Q (np.ndarray): Probability distribution of constellation points, shape (M,).
        pi_matrix (np.ndarray): Matrix of quadrature weights products, shape (M, K).
        w_matrix (np.ndarray): Matrix W (or g_matrix) of function values, shape (M, K).
        rho (float): Value of ρ parameter, 0 ≤ ρ ≤ 1.
        
    Returns:
        float: The computed first derivative dE0.
    """
    # Precomputed terms
    s = 1 / (1 + rho)
    sp = -1 / (1 + rho)**2

    # Compute log(W) for numerical stability (elementwise)
    log_W = np.log(w_matrix)
    
    # Compute π .* W^{-sρ}
    pi_term = pi_matrix * np.exp(-s * rho * log_W)
    
    # Compute Q^T * W^s using a logarithmic approach.
    # Here, inner_exp is computed elementwise as exp(s * log(W))
    inner_exp = np.exp(s * log_W)
    # np.dot(Q, inner_exp) simulates Q' * inner_exp (resulting in a 1D array of length K)
    Q_inner = np.dot(Q, inner_exp)  # shape: (K,)
    
    # Compute (Q^T * W^s)^ρ in a numerically stable manner.
    log_Q_inner = np.log(Q_inner)  # shape: (K,)
    outer_exp = np.exp(rho * log_Q_inner)  # shape: (K,)
    # Reshape to a column vector (K, 1) for proper matrix multiplication later.
    outer_term = outer_exp.reshape(-1, 1)
    
    # Final computation of F0 = Q' * pi_term * outer_term
    F0 = np.dot(Q, np.dot(pi_term, outer_term))  # scalar
    
    # First derivative components.
    # term1_part is computed elementwise over the matrix dimensions.
    term1_part = (-sp * rho * log_W) - (s * log_W)  # shape: (M, K)
    Term1 = np.dot(Q, np.dot(pi_term * term1_part, outer_term))  # scalar
    
    # Compute inner_exp_logW = inner_exp .* log_W (elementwise)
    inner_exp_logW = inner_exp * log_W  # shape: (M, K)
    # Compute Q' * inner_exp_logW, yielding a 1D array of shape (K,)
    Q_inner_log = np.dot(Q, inner_exp_logW)
    
    # Derivative of the outer term in a logarithmic manner.
    derivative_outer_part = log_Q_inner + rho * (sp * Q_inner_log / Q_inner)  # shape: (K,)
    
    # Term2: Q' * pi_term * (outer_exp .* derivative_outer_part)', where the product is elementwise.
    Term2 = np.dot(Q, np.dot(pi_term, (outer_exp * derivative_outer_part).reshape(-1, 1)))  # scalar
    
    # Sum the derivative components.
    F0_prime = Term1 + Term2
    
    # Compute the derivative dE0 using base-2 logarithm conversion.
    dE0 = -F0_prime / (F0 * np.log(2))
    
    return dE0