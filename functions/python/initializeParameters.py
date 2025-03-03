import numpy as np
from functions.python import generatePAMConstellation
from functions.python.GaussHermite_Locations_Weights import GaussHermite_Locations_Weights

def initializeParameters():
    """
    Initializes the parameters for E0 computation.
    
    Returns:
        params (dict): Dictionary containing:
            M      : Constellation size (int)
            SNR    : Signal-to-Noise Ratio (float)
            num_points : Number of points for evaluating ρ (int)
            N      : Number of quadrature nodes (int)
            nodes  : Quadrature nodes (np.ndarray of shape [N])
            weights: Quadrature weights (np.ndarray of shape [N])
            X      : Normalized constellation points (np.ndarray of shape [M])
            Q      : Probability distribution for constellation points (np.ndarray of shape [M])
    """
    params = {}
    # Initialize basic parameters
    params["M"] = 2           # Constellation size
    params["SNR"] = 1         # Signal-to-Noise Ratio
    params["num_points"] = 1000  # Number of points for ρ evaluation
    params["N"] = 2           # Number of quadrature nodes

    # Generate Gaussian-Hermite quadrature nodes and weights.
    # This function is assumed to be defined elsewhere.
    params["nodes"], params["weights"] = GaussHermite_Locations_Weights(params["N"])

    # Display quadrature information
    print("=" * 64)
    print("                 Gaussian-Hermite Quadrature")
    print("-" * 64)
    print("Quadrature Nodes:   ", np.array_str(params["nodes"], precision=4))
    print("Quadrature Weights: ", np.array_str(params["weights"], precision=4))
    print("Dimensions:         {} nodes".format(params["N"]))
    print("-" * 64)

    # Generate constellation points.
    # generatePAMConstellation is assumed to be defined elsewhere.
    X = generatePAMConstellation(params["M"], 2)
    avg_energy = np.mean(X**2)
    params["X"] = X / np.sqrt(avg_energy)

    # Display constellation information
    print("                    Constellation Points")
    print("-" * 64)
    print("Points:             ", np.array_str(params["X"], precision=4))
    print("Dimensions:         {} points".format(len(params["X"])))
    print("Average Energy:     {:.4f} (normalized)".format(np.mean(params["X"]**2)))
    print("-" * 64)

    # Generate and display probability distribution (uniform distribution)
    params["Q"] = np.ones(params["M"]) / params["M"]
    print("                  Probability Distribution")
    print("-" * 64)
    print("Distribution:       ", np.array_str(params["Q"], precision=4))
    print("Dimensions:         {} probabilities".format(len(params["Q"])))
    print("Sum of Prob:        {:.4f}".format(np.sum(params["Q"])))
    print("=" * 64)
    
    return params