import datetime
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

import numpy as np
import time

# -------------------- System Parameters --------------------
nodesN = 20          # Number of quadrature nodes (for Gauss-Hermite integration)
signalNoiseRatio = 1        # Signal-to-Noise Ratio
constellationM = 4           # Size of the PAM constellation (number of symbols)
d = 2           # Distance parameter (spacing between constellation points)
transmissionRate = 0.5         # Communication rate

# -------------------- Compute Gauss-Hermite Nodes and Weights --------------------
# These are used for numerical integration over a Gaussian density.
nodes, weights = GaussHermite_Locations_Weights(nodesN)
# print(f"nodes = {nodes}")
# print(f"weights = {weights}")

# -------------------- Generate and Normalize PAM Constellation --------------------
# Create a PAM constellation with M symbols and spacing d.
X = generatePAMConstellation(constellationM, d)

# -------------------- Initialize the Probability Distribution --------------------
# Create a 1D array with equal probability for each symbol.
Q = np.ones(constellationM) / constellationM

# -------------------- Define the Gaussian Function --------------------
# G(z) = exp(-|z|^2) / pi
G = lambda z: np.exp(-np.abs(z)**2) / np.pi

# -------------------- Create Required Matrices for Computation --------------------
pi_matrix = createPiMatrix(constellationM, nodesN, weights)
z_matrix = createComplexNodesMatrix(nodes)
g_matrix = createGMatrix(X, z_matrix, signalNoiseRatio, G)

# -------------------- Find an Initial Guess for the Optimization --------------------
# Compute E0 and its derivatives at rho = 0 and rho = 1.
E00 = 0  # E0 at rho = 0
E01 = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix)  # E0 at rho = 1
E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0)    # Derivative at rho = 0
E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1)    # Derivative at rho = 1

# Use Hermite interpolation to get an initial candidate for rho.
rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, transmissionRate)

# -------------------- Set Optimization Parameters --------------------
tol = 1e-10   # Convergence tolerance

# -------------------- Use Newton's Method to Find the Optimal rho --------------------
# The goal is to maximize E0(rho) - rho*R.
start_overall = time.perf_counter()

if transmissionRate > E0P0:
    # If the communication rate is higher than the derivative at 0, the maximum is at rho = 0.
    optimalRho = 0
    errorExponent = 0
elif transmissionRate < E0P1:
    # If the communication rate is lower than the derivative at 1, the maximum is at rho = 1.
    optimalRho = 1
    errorExponent = computeEoForRhoExponential(optimalRho, Q, pi_matrix, g_matrix) - transmissionRate
else:
    # Otherwise, use Newton's method starting from rho_star to find the optimum.
    start_newton = time.perf_counter()
    optimalRho, errorExponent = optimizationNewton(Q, pi_matrix, g_matrix, transmissionRate, tol, rho_star)
    errorExponent = errorExponent - optimalRho * transmissionRate
    newton_time = time.perf_counter() - start_newton

overall_time = time.perf_counter() - start_overall

optimalRho = np.real(optimalRho)
errorExponent = np.real(errorExponent)

# -------------------- Display the Results --------------------
print(f"optimalRho = {optimalRho:.10f}")
print(f"errorExponent = {errorExponent:.10f}")
print(f"Optimisation time: {overall_time * 1000:.10f} ms")

tableName = 'exponents'
simulationId = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

"""
# Call the function directly with the parameters.
try:
    add_data(
        table=tableName,
        constellationM=constellationM,
        simulationId=simulationId,
        nodesN=nodesN,
        SNR=signalNoiseRatio,
        transmissionRate=transmissionRate,
        errorExponent=errorExponent,
        optimalRho=optimalRho
    )
except Exception as e:
    print("Error inserting data!")
    print(e)
"""