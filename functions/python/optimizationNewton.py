import numpy as np
from functions.python.derivativesE0 import derivativesE0

def optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star):
    """
    Optimize the E0 function using Newton's method.

    Parameters:
        Q         : 1D array or similar data structure required for computing E0.
        pi_matrix : 2D array of probability-related values for E0.
        g_matrix  : 2D array containing gain values or similar parameters for E0.
        R         : Scalar parameter representing a rate or threshold used in the optimization.
        tol       : Tolerance for the convergence criterion based on the change in rho.
        rho_star  : Initial guess for the value of rho.

    Returns:
        rho_opt   : The optimized value of rho at which E0 is maximized.
        E0_max    : The value of the function E0 evaluated at rho_opt.
    """
    iter = 1
    log2_const = np.log(2)
    rho = rho_star
    max_iter = 100
    tol_curvature = 1e-10

    while iter < max_iter:
        # Compute F0, dF0, d2F0 from derivativesE0 function.
        F0, dF0, d2F0 = derivativesE0(Q, pi_matrix, g_matrix, rho)
        E0 = -np.log2((1/np.pi) * F0)

        # Precompute repeated terms to optimize divisions.
        f_inv = 1 / F0
        fp_f = dF0 * f_inv

        # Compute gradient and curvature.
        gradient = (-fp_f / log2_const) - R
        curvature = -((d2F0 * f_inv) - (fp_f * fp_f)) / log2_const

        # Convergence check based on the Newton update or small curvature.
        if abs(gradient / curvature) < tol or abs(curvature) < tol_curvature:
            break

        # Compute new rho value using Newton's update.
        rho_new = rho - gradient / curvature

        # Check convergence based on the change in rho.
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break

        rho = rho_new
        iter += 1

    rho_opt = rho
    E0_max = E0

    print(f"rho_opt = {rho_opt:.6f}")
    print(f"E0_max = {E0_max:.6f}")
    print(f"iter = {iter}")
    
    return rho_opt, E0_max