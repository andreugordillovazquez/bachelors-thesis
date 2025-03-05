#!/usr/bin/env python3
import datetime
import numpy as np
import time
import uuid
from functions.add_data import add_data
from functions.python.GaussHermite_Locations_Weights import GaussHermite_Locations_Weights
from functions.python.generatePAMConstellation import generatePAMConstellation
from functions.python.createPiMatrix import createPiMatrix
from functions.python.createComplexNodesMatrix import createComplexNodesMatrix
from functions.python.createGMatrix import createGMatrix
from functions.python.computeEoForRhoExponential import computeEoForRhoExponential
from functions.python.computeFirstDerivativeE0 import computeFirstDerivativeE0
from functions.python.demoHermiteInterpolation import demoHermiteInterpolation
from functions.python.optimizationNewton import optimizationNewton

def run_simulation(constellationM, signalNoiseRatio, transmissionRate, nodesN=20):
    """
    Run a single simulation with the given parameters.
    
    Args:
        constellationM (int): Constellation size
        signalNoiseRatio (float): SNR in dB
        transmissionRate (float): Transmission rate
        nodesN (int): Number of quadrature nodes
        
    Returns:
        tuple: (optimalRho, errorExponent, execution_time_ms)
    """
    
    # Distance parameter (spacing between constellation points)
    d = 2
    
    # Start timing
    start_overall = time.perf_counter()
    
    # Compute Gauss-Hermite Nodes and Weights
    nodes, weights = GaussHermite_Locations_Weights(nodesN)
    
    # Generate and Normalize PAM Constellation
    X = generatePAMConstellation(constellationM, d)
    
    # Initialize the Probability Distribution
    Q = np.ones(constellationM) / constellationM
    
    # Define the Gaussian Function
    G = lambda z: np.exp(-np.abs(z)**2) / np.pi
    
    # Create Required Matrices for Computation
    pi_matrix = createPiMatrix(constellationM, nodesN, weights)
    z_matrix = createComplexNodesMatrix(nodes)
    g_matrix = createGMatrix(X, z_matrix, signalNoiseRatio, G)
    print(g_matrix)
    
    """# Save g_matrix to CSV (real parts only)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"g_matrix_M{constellationM}_SNR{signalNoiseRatio}_N{nodesN}_{timestamp}.csv"
    
    # Save only real parts
    np.savetxt(
        filename,
        g_matrix.real,  # Only real parts
        delimiter=',',
        fmt='%.10e'
    )
    print(f"G matrix (real parts) saved to {filename}")"""
    
    # Find an Initial Guess for the Optimization
    E00 = 0  # E0 at rho = 0
    E01 = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix)  # E0 at rho = 1
    E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0)    # Derivative at rho = 0
    E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1)    # Derivative at rho = 1
    
    # Use Hermite interpolation to get an initial candidate for rho
    rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, transmissionRate)
    
    # Set Optimization Parameters
    tol = 1e-7   # Convergence tolerance
    
    # Use Newton's Method to Find the Optimal rho
    if transmissionRate > np.real(E0P0):
        # If the communication rate is higher than the derivative at 0, the maximum is at rho = 0
        optimalRho = 0
        errorExponent = 0
    elif transmissionRate < np.real(E0P1):
        # If the communication rate is lower than the derivative at 1, the maximum is at rho = 1
        optimalRho = 1
        errorExponent = np.real(computeEoForRhoExponential(optimalRho, Q, pi_matrix, g_matrix) - transmissionRate)
    else:
        # Otherwise, use Newton's method starting from rho_star to find the optimum
        optimalRho, E0_max = optimizationNewton(Q, pi_matrix, g_matrix, transmissionRate, tol, rho_star)
        errorExponent = np.real(E0_max - optimalRho * transmissionRate)
    
    # Ensure results are real numbers
    optimalRho = np.real(optimalRho)
    errorExponent = np.real(errorExponent)
    
    # Calculate execution time
    execution_time_ms = (time.perf_counter() - start_overall) * 1000
    
    return optimalRho, errorExponent, execution_time_ms

def main():
    # Fixed parameters
    constellationM = 4
    nodesN = 20
    tableName = 'exponents'
    
    # SNR range from -10 to 20 dB
    # snr_values_dB = np.arange(5, 10, 1)  # Step size of 2 dB for faster testing
    snr_values_dB = [1]
    
    # Transmission rate range from 0 to 1 with step size 0.05
    # rate_values = np.arange(0, 2.01, 0.1) 
    rate_values = [0.5]
    
    # Total number of simulations
    total_simulations = len(snr_values_dB) * len(rate_values)
    completed = 0
    
    # Store all results for later analysis
    all_results = []
    
    print(f"Starting {total_simulations} simulations...")
    print(f"Constellation size (M): {constellationM}")
    print(f"Number of quadrature nodes (N): {nodesN}")
    print(f"SNR range: {snr_values_dB[0]} to {snr_values_dB[-1]} dB")
    print(f"Rate range: {rate_values[0]} to {rate_values[-1]}")
    print("-" * 60)
    
    # Run simulations for each combination of SNR and rate
    for snr_dB in snr_values_dB:
        for rate in rate_values:
            # Generate a unique simulation ID
            simulationId = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%s')}"
            
            print(f"Running simulation {completed+1}/{total_simulations}:")
            print(f"  SNR: {snr_dB} dB, Rate: {rate}")
            
            # Run the simulation
            optimalRho, errorExponent, execution_time = run_simulation(
                constellationM=constellationM,
                signalNoiseRatio=snr_dB,
                transmissionRate=rate,
                nodesN=nodesN
            )
            
            # Print results
            print(f"  Results: ρ* = {optimalRho:.6f}, E0 = {errorExponent:.6f}")
            print(f"  Execution time: {execution_time:.2f} ms")
            
            # Store results
            result = {
                'constellationM': constellationM,
                'simulationId': simulationId,
                'nodesN': nodesN,
                'SNR': snr_dB,
                'transmissionRate': float(f"{rate:.15f}"),
                'errorExponent': float(f"{errorExponent:.15f}"),
                'optimalRho': float(f"{optimalRho:.15f}"),
                'execution_time_ms': execution_time
            }
            all_results.append(result)
            """
            # Comment out DynamoDB upload
            try:
                add_data(
                    table=tableName,
                    constellationM=constellationM,
                    simulationId=simulationId,
                    nodesN=nodesN,
                    SNR=snr_dB,  # Store SNR in dB
                    transmissionRate=float(f"{rate:.15f}"),
                    errorExponent=float(f"{errorExponent:.15f}"),
                    optimalRho=float(f"{optimalRho:.15f}")
                )
                print(f"  Data uploaded to DynamoDB successfully.")
            except Exception as e:
                print(f"  Error uploading to DynamoDB: {e}")
            """
            completed += 1
            print("-" * 60)
    
    print(f"All {total_simulations} simulations completed!")
    
    # Display summary of results
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'SNR (dB)':<10} {'Rate':<10} {'Error Exponent':<15} {'Optimal ρ':<15}")
    print("-" * 60)
    
    # Group results by SNR for better readability
    for snr_dB in snr_values_dB:
        snr_results = [r for r in all_results if r['SNR'] == snr_dB]
        for result in snr_results:
            print(f"{result['SNR']:<10.1f} {result['transmissionRate']:<10.4f} {result['errorExponent']:<15.6f} {result['optimalRho']:<15.6f}")
        print("-" * 60)

if __name__ == "__main__":
    main() 