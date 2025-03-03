import numpy as np
import matplotlib.pyplot as plt

def plotE0(rho_values, Eo_rho, params):
    """
    Visualizes how E0(ρ) changes as we vary ρ.
    
    Parameters:
        rho_values (array-like): Range of ρ values from 0 to 1.
        Eo_rho (array-like): Computed E0 values corresponding to each ρ.
        params (dict): Dictionary containing system parameters. Expected to include the key 'nodes'.
    
    The plot shows:
        - x-axis: ρ values from 0 to 1,
        - y-axis: E0(ρ) values.
    """
    # Close any existing figures.
    plt.close("all")
    
    # Create a new figure.
    plt.figure()
    
    # Plot E0(ρ) with a thick line.
    plt.plot(rho_values, Eo_rho, linewidth=2)
    
    # Create a title that includes the number of quadrature nodes.
    num_nodes = len(params["nodes"]) if "nodes" in params else "unknown"
    plt.title(f"E₀(ρ) using {num_nodes} quadrature nodes")
    
    # Label axes and add a grid.
    plt.xlabel("ρ")
    plt.ylabel("E₀(ρ)")
    plt.grid(True)
    
    # Optional: Print key analysis points to the console.
    print("=" * 64)
    print("                       E0(ρ) Plot Analysis")
    print("-" * 64)
    print(f"Maximum E0 value: {np.max(Eo_rho):.4f}")
    print(f"E0 at ρ = 0: {Eo_rho[0]:.4f}")
    print(f"E0 at ρ = 1: {Eo_rho[-1]:.4f}")
    print("=" * 64)
    
    # Display the plot.
    plt.show()