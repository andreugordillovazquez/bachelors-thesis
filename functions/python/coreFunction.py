import numpy as np
import time

def coreFunction(M, n, SNR, R):
    """
    Computes:
      1) The total estimated running time (in ms) for all iterations,
         based on an empirical factor (in ns) per (n^2 * M^2).
      2) The actual running time in seconds.
      3) The running times for each key section, stored as a list of dictionaries.
    
    Parameters:
        M (list or array-like): Constellation sizes (scalar or vector).
        n (list or array-like): Number of Gauss-Hermite nodes per real part (scalar or vector).
        SNR (float): Signal-to-noise ratio.
        R (float): Transmission rate.
        
    Returns:
        totalEstimatedRunningTime (float): Estimated total running time (ms).
        totalRunningTime (float): Actual total running time (seconds).
        runningTimes (list): List of dictionaries with timing info for each (M, n) iteration.
    """
    overallTimer = time.perf_counter()  # Start overall timing
    
    # Ensure M and n are numpy arrays for consistent operations
    M_arr = np.array(M)
    n_arr = np.array(n)
    
    print("Simulation Parameters:")
    print(f"  SNR  = {SNR:.2f}")
    print(f"  Rate = {R:.2f}")
    print(f"  M values: [{', '.join(map(str, M_arr))}]")
    print(f"  n values: [{', '.join(map(str, n_arr))}]")
    print("--------------------------------------------------------")
    
    # Total Estimated Running Time Calculation
    # Empirical factor from Excel (in nanoseconds per (n^2 * M^2))
    ratio = 112.23  # ns
    totalEstimatedRunningTime = (ratio * np.sum(n_arr**2) * np.sum(M_arr**2)) / 1e6  # convert ns to ms
    print(f"Total estimated running time over all iterations: {totalEstimatedRunningTime:.2f} ms")
    print("--------------------------------------------------------")
    
    totalComb = len(M_arr) * len(n_arr)
    iterCount = 0
    runningTimes = []  # List to hold timing data for each iteration
    
    # Loop for each constellation size M
    for m_idx, M_val in enumerate(M_arr, start=1):
        print(f"\n================ Processing M = {M_val} ({m_idx} of {len(M_arr)}) ================")
        
        # M-dependent setup: Generate and normalize the PAM constellation
        tM_start = time.perf_counter()
        d = 2  # Define PAM spacing
        print(f"-> Generating PAM constellation with M = {M_val} symbols and spacing d = {d}...")
        X = generatePAMConstellation(M_val, d)
        # Normalize: X = X / sqrt(mean(X.^2))
        X = X / np.sqrt(np.mean(X**2))
        print("   Constellation generated and normalized.")
        
        # Initialize the probability distribution for constellation symbols (length M_val)
        Q = np.full(M_val, 1 / M_val)
        print("-> Probability distribution for constellation symbols initialized.")
        tM_elapsed = (time.perf_counter() - tM_start) * 1000  # in ms
        print(f"M-dependent setup time: {tM_elapsed:.6f} ms")
        
        # Inner Loop: For each number of Gauss-Hermite nodes n
        for n_idx, n_val in enumerate(n_arr, start=1):
            iterCount += 1
            print(f"\n---------------- Iteration {iterCount} of {totalComb} ----------------")
            print(f"Processing for n = {n_val}, M = {M_val}")
            
            # n-dependent setup: Compute Gauss-Hermite nodes, weights, and complex nodes matrix
            tN_start = time.perf_counter()
            N = n_val  # Number of Gauss-Hermite nodes
            print(f"-> Computing Gauss-Hermite nodes and weights with N = {N}...")
            nodes, weights = GaussHermite_Locations_Weights(N)
            print("   Gauss-Hermite nodes and weights computed.")
            
            # Compute the complex nodes matrix (depends only on n)
            z_matrix = createComplexNodesMatrix(nodes)
            print(f"-> z_matrix computed for N = {N}.")
            tN_elapsed = (time.perf_counter() - tN_start) * 1000  # in ms
            print(f"n-dependent setup time: {tN_elapsed:.6f} ms")
            
            # Combined M & n dependent computations: Create matrices for computation
            tCombined_start = time.perf_counter()
            print("-> Creating required matrices for computation...")
            # Define the Gaussian function G
            G = lambda z: np.exp(-np.abs(z)**2) / np.pi
            # pi_matrix depends on both M and n
            pi_mat = createPiMatrix(M_val, N, weights)
            # g_matrix depends on the constellation X and z_matrix
            g_mat = createGMatrix(X, z_matrix, SNR, G)
            tCombined_elapsed = (time.perf_counter() - tCombined_start) * 1000  # in ms
            print(f"Combined (M & n) computation time: {tCombined_elapsed:.6f} ms")
            
            # Optimization routine: Find the Optimal rho
            tOpt_start = time.perf_counter()
            print(f"-> Starting optimization for n = {n_val}, M = {M_val}...")
            # Compute E0 at endpoints (rho = 0 and rho = 1) and their derivatives
            E00  = 0  # E0 at rho = 0
            E01  = computeEoForRhoExponential(1, Q, pi_mat, g_mat)  # E0 at rho = 1
            E0P0 = computeFirstDerivativeE0(Q, pi_mat, g_mat, 0)
            E0P1 = computeFirstDerivativeE0(Q, pi_mat, g_mat, 1)
            print(f"   E0 at rho=0: {E00:.6f}, E0 at rho=1: {E01:.6f}")
            print(f"   Derivative at rho=0: {E0P0:.6f}, Derivative at rho=1: {E0P1:.6f}")
            
            # Obtain an initial guess for rho using Hermite interpolation
            print("-> Performing Hermite interpolation for an initial guess of rho...")
            rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, R)
            print(f"   Initial guess for rho: {rho_star:.6f}")
            
            tol = 1e-10  # Convergence tolerance
            if R > E0P0:
                print(f"   R ({R:.6f}) > derivative at 0 ({E0P0:.6f}): setting optimal rho = 0")
                rho_opt = 0
                E0_max  = E00
            elif R < E0P1:
                print(f"   R ({R:.6f}) < derivative at 1 ({E0P1:.6f}): setting optimal rho = 1")
                rho_opt = 1
                E0_max  = E01
            else:
                print("-> Running Newton's method to optimize rho...")
                rho_opt, E0_max = optimizationNewton(Q, pi_mat, g_mat, R, tol, rho_star)
            
            print("   Optimization complete.")
            print(f"   Optimal rho: {rho_opt:.10f}, Maximum E0: {E0_max:.10f}")
            maxER = E0_max - rho_opt * R
            print(f"   Maximum E(R): {maxER:.10f}")
            tOpt_elapsed = (time.perf_counter() - tOpt_start) * 1000  # in ms
            print(f"Optimization routine time: {tOpt_elapsed:.6f} ms")
            
            # Store the running times in the list
            runningTimes.append({
                'M_val': M_val,
                'n_val': n_val,
                'M_setup': tM_elapsed,
                'n_setup': tN_elapsed,
                'Combined': tCombined_elapsed,
                'Optimization': tOpt_elapsed
            })
    
    print("\n==================== END OF SIMULATIONS ====================")
    totalRunningTime = time.perf_counter() - overallTimer  # in seconds
    
    return totalEstimatedRunningTime, totalRunningTime, runningTimes