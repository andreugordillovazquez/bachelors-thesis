import numpy as np
import time
import matplotlib.pyplot as plt

def performanceTest(params):
    """
    Evaluates how computation time scales with constellation size.
    
    This function tests a range of constellation sizes by:
      - Generating normalized constellation points,
      - Computing the necessary matrices,
      - Evaluating the E0 function over a range of ρ values,
      - Measuring the execution time,
      - Displaying timing information, and
      - Plotting both linear and log-scale performance graphs.
      
    Parameters:
        params (dict): Base configuration dictionary containing quadrature settings.
                       Expected keys include: 'nodes', 'weights', 'SNR', 'num_points', etc.
    """
    # Define constellation sizes to test.
    constellation_sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1028])
    execution_times = np.zeros(len(constellation_sizes))
    constellation_points = [None] * len(constellation_sizes)
    
    # Test each constellation size.
    for idx, M_const in enumerate(constellation_sizes):
        params["M_const"] = int(M_const)
        
        # Generate normalized constellation points using linspace.
        X = np.linspace(-3, 3, params["M_const"])
        avg_energy = np.mean(X**2)
        X = X / np.sqrt(avg_energy)
        params["X"] = X
        constellation_points[idx] = X.copy()
        
        # Set equal probabilities for all points.
        params["Q"] = np.ones(params["M_const"]) / params["M_const"]
        
        # Time the E0 computation.
        start_time = time.perf_counter()
        
        # Create necessary matrices.
        z_matrix = createComplexNodesMatrix(params["nodes"])
        pi_matrix = createPiMatrix(params["M_const"], len(params["nodes"]), params["weights"])
        # Define the Gaussian function as a lambda.
        g_matrix = createGMatrix(params["X"], z_matrix, params["SNR"],
                                 lambda z: (1 / np.pi) * np.exp(-np.abs(z)**2))
        
        # Compute E0 for all ρ values.
        rho_values = np.linspace(0, 1, params["num_points"])
        Eo_rho = np.zeros_like(rho_values)
        for i, rho in enumerate(rho_values):
            Eo_rho[i] = computeEoForRho(rho, params["Q"], pi_matrix, g_matrix)
        
        exec_time = time.perf_counter() - start_time
        execution_times[idx] = exec_time
        
        # Display progress with timing information.
        print(f"M = {params['M_const']} points:")
        print(f"    Execution time: {exec_time:.3f} seconds")
    
    # Display summary statistics.
    print("----------------------------------------------------------------")
    print("Performance Summary:")
    print(f"Fastest run: M = {constellation_sizes[0]} points ({execution_times[0]:.3f} s)")
    print(f"Slowest run: M = {constellation_sizes[-1]} points ({execution_times[-1]:.3f} s)")
    print("================================================================")
    
    # Create performance visualizations.
    plt.close("all")  # Ensure clean plotting.
    
    # Create a figure with two subplots side by side.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale plot (left subplot).
    axs[0].plot(constellation_sizes, execution_times, linewidth=2, marker="o",
                markersize=8)
    axs[0].set_title("Linear Scale: Computation Time vs. Constellation Size")
    axs[0].set_xlabel("Constellation Size (M points)")
    axs[0].set_ylabel("Computation Time (seconds)")
    axs[0].grid(True)
    
    # Logarithmic scale plot (right subplot).
    axs[1].loglog(constellation_sizes, execution_times, linewidth=2, marker="o",
                  markersize=8)
    axs[1].set_title("Log Scale: Computation Time vs. Constellation Size")
    axs[1].set_xlabel("Constellation Size (M points)")
    axs[1].set_ylabel("Computation Time (seconds)")
    axs[1].grid(True)
    
    # Add an overall title.
    fig.suptitle("E₀ Computation Performance Analysis", fontsize=14)
    
    # Compute and display the approximate complexity.
    # Fit a line to the log10-scaled data.
    p = np.polyfit(np.log10(constellation_sizes), np.log10(execution_times), 1)
    complexity_order = p[0]
    print(f"Approximate computational complexity: O(n^{complexity_order:.2f})")
    
    plt.show()