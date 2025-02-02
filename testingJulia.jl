using LinearAlgebra
using Plots
using Printf
using BenchmarkTools

# Hardcoded Gauss-Hermite nodes and weights for N = 2
function GaussHermite_Locations_Weights(N)
    if N == 2
        nodes = [-1 / sqrt(2), 1 / sqrt(2)]
        weights = [sqrt(pi) / 2, sqrt(pi) / 2]
        return nodes, weights
    else
        error("Only N = 2 is supported with hardcoded values.")
    end
end

# Function to generate PAM constellation
function generatePAMConstellation(M, d)
    return [(2 * i - M - 1) * d / 2 for i in 1:M]
end

# Constants
Array_M = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N = 2
nodes, weights = GaussHermite_Locations_Weights(N)

SNR = 1
rho_values = [0.5]
computation_times = zeros(Float64, length(Array_M))

# Gaussian function
G(z) = (1 / π) * exp(-abs2(z))

Eo_rho = zeros(Float64, length(rho_values))

println("Array_M: $Array_M")
println("Nodes: $nodes")
println("Weights: $weights")
println("Signal-to-Noise Ratio (SNR): $SNR")
println("Rho values: $rho_values")

# Main loop
for (i, M) in enumerate(Array_M)
    d = 2

    # Generate and normalize constellation
    X = generatePAMConstellation(M, d)
    avg_energy = mean(x^2 for x in X)
    X .= X ./ sqrt(avg_energy)

    # Equal probabilities
    Q = fill(1.0 / M, M)

    # Measure computation time
    t_start = @elapsed begin
        for (j, rho) in enumerate(rho_values)
            integral_sum = 0.0

            # Loop over PAM symbols
            for (x_idx, x) in enumerate(X)
                quadrature_sum = 0.0

                # Nested loops over quadrature nodes
                for k in 1:N
                    for l in 1:N
                        z = nodes[k] + im * nodes[l]
                        fp_sum = 0.0

                        for x_bar in X
                            fp_sum += Q[x_idx] * (G(z + sqrt(SNR) * x - sqrt(SNR) * x_bar)^(1 / (1 + rho)))
                        end

                        fp = fp_sum / (G(z)^(1 / (1 + rho)))
                        quadrature_sum += weights[k] * weights[l] * fp^rho
                    end
                end

                integral_sum += Q[x_idx] * quadrature_sum
            end

            Eo_rho[j] = -log2((1 / π) * integral_sum)
        end
    end

    computation_times[i] = round(t_start * 1000)  # Convert seconds to milliseconds
    println("Computation time for M = $M: $(round(computation_times[i], digits=2)) ms")
end