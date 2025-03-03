def hermiteInterp(A, B, C, D):
    """
    Constructs a cubic Hermite interpolant P(rho) on [0, 1] that satisfies:
        P(0)  = A,   P'(0) = B,
        P(1)  = C,   P'(1) = D.
    
    The cubic polynomial is defined as:
        P(rho) = a + b*rho + c*rho^2 + d*rho^3,
    with coefficients:
        a = A,
        b = B,
        c = 3*(C - A) - 2*B - D,
        d = -2*(C - A) + B + D.
    
    Parameters:
        A (float): Value of P(0).
        B (float): Value of P'(0).
        C (float): Value of P(1).
        D (float): Value of P'(1).
        
    Returns:
        p (function): A lambda function representing P(rho).
        dp (function): A lambda function representing P'(rho).
    """
    a = A
    b = B
    c = 3*(C - A) - 2*B - D
    d = -2*(C - A) + B + D

    p  = lambda rho: a + b * rho + c * (rho**2) + d * (rho**3)
    dp = lambda rho: b + 2 * c * rho + 3 * d * (rho**2)
    
    return p, dp