import numpy as np

def hermiteInterp(A, B, C, D):
    """
    Compute cubic Hermite interpolation coefficients and functions.
    
    Args:
        A: G(0) value
        B: G'(0) value
        C: G(1) value
        D: G'(1) value
        
    Returns:
        tuple: (p, dp, coefs) where:
            p: Function handle for the polynomial
            dp: Function handle for its derivative
            coefs: Array of coefficients [a, b, c, d]
    """
    # Extract real parts if complex
    A = np.real(A)
    B = np.real(B)
    C = np.real(C)
    D = np.real(D)
    
    # Print input values for debugging
    print("Input values:", A, B, C, D)
    
    # Compute the cubic coefficients
    a = A
    b = B
    c = 3*(C - A) - 2*B - D
    d = -2*(C - A) + B + D
    
    # Print coefficients for debugging
    print("Computed coefficients:", a, b, c, d)
    
    # Create function handles for the polynomial and its derivative
    def p(rho):
        return a + b*rho + c*rho**2 + d*rho**3
    
    def dp(rho):
        return b + 2*c*rho + 3*d*rho**2
    
    coefs = np.array([a, b, c, d])
    return p, dp, coefs

def demoHermiteInterpolation(E0_0, E0p_0, E0_1, E0p_1, R):
    """
    Demo of constructing G(rho) = E0(rho) - rho*R with known values/derivatives
    at rho=0 and rho=1, then doing Hermite interpolation to approximate the maximum.
    
    Args:
        E0_0: E0(0) value
        E0p_0: E0'(0) value
        E0_1: E0(1) value
        E0p_1: E0'(1) value
        R: Rate value
        
    Returns:
        float: rho_star, the value of rho that maximizes G(rho)
    """
    # Extract real parts if complex
    if isinstance(E0p_0, np.ndarray):
        E0p_0 = E0p_0[0]  # Extract first element if array
    if isinstance(E0p_1, np.ndarray):
        E0p_1 = E0p_1[0]  # Extract first element if array
        
    # Print input values for debugging
    print("Input values to demoHermiteInterpolation:", E0_0, E0p_0, E0_1, E0p_1, R)
    
    # Set known endpoints and derivative values for G(rho) = E0(rho) - rho*R
    A = E0_0
    B = E0p_0 - R
    C = E0_1 - R
    D = E0p_1 - R
    
    # Build the cubic Hermite interpolant for G(rho)
    p, dp, coefs = hermiteInterp(A, B, C, D)
    
    # Find critical points in [0,1]
    # Solve: 3*d*rho^2 + 2*c*rho + b = 0
    roots_dp = np.roots([3*coefs[3], 2*coefs[2], coefs[1]])
    
    # Filter for real roots in [0,1]
    mask = (np.abs(np.imag(roots_dp)) < 1e-10) & (np.real(roots_dp) >= 0) & (np.real(roots_dp) <= 1)
    roots_dp = np.real(roots_dp[mask])
    
    # Evaluate p at endpoints and those roots
    candidates = np.array([0.0, 1.0] + list(roots_dp))
    p_vals = np.array([p(x) for x in candidates])
    idx = np.argmax(p_vals)
    rho_star = candidates[idx]
    p_max = p_vals[idx]
    
    print(f'Approx. max of G(rho) is {p_max:.6f} at rho = {rho_star:.4f}')
    
    return rho_star