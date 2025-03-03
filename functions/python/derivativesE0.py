import numpy as np

def derivativesE0(Q, pi_matrix, w_matrix, rho):
    """
    Computes F0, its first derivative dF0, and its second derivative d2F0
    based on the inputs Q, pi_matrix, w_matrix, and parameter rho.
    
    Parameters:
        Q         : 1D array (length M) representing the probability vector.
        pi_matrix : 2D array of quadrature weight products.
        w_matrix  : 2D array (same shape as pi_matrix) representing W.
        rho       : Scalar parameter.
        
    Returns:
        F0   : Scalar, the computed value F0 = Q' * (pi_term) * (outer_term).
        dF0  : Scalar, the first derivative of F0.
        d2F0 : Scalar, the second derivative of F0.
    """
    # Precomputed constants
    s   = 1 / (1 + rho)
    sp  = -1 / (1 + rho)**2
    spp = 2 / (1 + rho)**3
    
    # Compute log(W) for numerical stability
    logW = np.log(w_matrix)
    
    # Compute the "pi_term": pi_matrix .* W^{-s*rho}
    pi_term = pi_matrix * np.exp(-s * rho * logW)
    
    # Compute inner exponentials: W^s and then Q' * W^s
    inner_exp   = np.exp(s * logW)  # same as w_matrix**s
    Q_inner     = np.dot(Q, inner_exp)  # shape: (L,), where L is the number of columns of inner_exp
    log_Q_inner = np.log(Q_inner)
    outer_exp   = np.exp(rho * log_Q_inner)
    # Make outer_term a column vector for subsequent matrix multiplications
    outer_term  = outer_exp.reshape(-1, 1)
    
    # Compute F0 = Q' * pi_term * outer_term
    # np.dot(pi_term, outer_term) yields a column vector; then dot with Q gives a scalar.
    F0 = np.dot(Q, np.dot(pi_term, outer_term).flatten())
    
    # === Compute first derivative dF0 ===
    # Precompute common factor (for term1)
    common_factor = sp * rho + s  # so that term1_part = -common_factor * logW
    term1_part = -common_factor * logW  # elementwise multiplication
    
    # Term 1: derivative contribution from the pi_term component
    Term1 = np.dot(Q, np.dot(pi_term * term1_part, outer_term).flatten())
    
    # Term 2: derivative contribution from the outer_term component
    inner_exp_logW = inner_exp * logW
    numerator = np.dot(Q, inner_exp_logW) * sp  # elementwise multiplication after dot product
    d_log_Q_inner = numerator / Q_inner  # elementwise division
    derivative_outer_part = log_Q_inner + rho * d_log_Q_inner
    d_outer_exp = outer_exp * derivative_outer_part
    d_outer_term = d_outer_exp.reshape(-1, 1)
    Term2 = np.dot(Q, np.dot(pi_term, d_outer_term).flatten())
    
    # Total first derivative
    dF0 = Term1 + Term2
    
    # === Compute second derivative d2F0 ===
    term1_sq = term1_part**2
    common_factor2 = spp * rho + 2 * sp  # derivative of (sp*rho+s)
    part2 = -common_factor2 * logW  # equivalent to (-spp*rho - 2*sp)*logW
    
    # Term 1: from the square of the first derivative term (pi_term component)
    Term1_2 = np.dot(Q, np.dot(pi_term * term1_sq, outer_term).flatten())
    
    # Term 2: from the second derivative of the pi_term component
    Term2_2 = np.dot(Q, np.dot(pi_term * part2, outer_term).flatten())
    
    # Term 3: cross term from derivative of outer_term
    Term3_2 = 2 * np.dot(Q, np.dot(pi_term * term1_part, d_outer_term).flatten())
    
    # Term 4: second derivative from the outer_term (square of derivative factor)
    Term4_2 = np.dot(Q, np.dot(pi_term, (outer_exp * (derivative_outer_part**2)).reshape(-1, 1)).flatten())
    
    # Term 5: additional second derivative contribution from outer_term
    numerator_component1 = np.dot(Q, inner_exp_logW)
    comp1 = 2 * sp * (numerator_component1 / Q_inner)
    logW_sq = logW**2
    combined = spp * logW + sp**2 * logW_sq
    numerator_component2 = np.dot(Q, inner_exp * combined)
    comp2 = rho * (numerator_component2 / Q_inner)
    comp3 = -rho * ((sp * (numerator_component1 / Q_inner))**2)
    Term5_2 = np.dot(Q, np.dot(pi_term, (outer_exp * (comp1 + comp2 + comp3)).reshape(-1, 1)).flatten())
    
    # Sum all terms to obtain d2F0
    d2F0 = Term1_2 + Term2_2 + Term3_2 + Term4_2 + Term5_2
    
    return F0, dF0, d2F0