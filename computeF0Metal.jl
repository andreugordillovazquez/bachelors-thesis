###############################################################################
# File: computeF0Metal.jl
#
# Usage:
#    1) julia computeF0Metal.jl
#    2) The script will compute F0, F0', F0'' for random data, using the Apple
#       GPU for log() and exp() via Metal.jl
###############################################################################

# Install required packages if not already installed
import Pkg
for pkg in ["Metal", "GPUArrays"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using Metal
using GPUArrays
using Random
using LinearAlgebra
using Statistics

###############################################################################
# computeF0_all_metal:
#
#  Computes F0(ρ, Q), F0'(ρ, Q), and F0''(ρ, Q) for the i.i.d. random-coding
#  error exponent, with partial GPU acceleration on Apple Silicon via Metal.jl.
#
# Inputs:
#   - rho     : scalar (Float32)
#   - Q       : M×1 Vector{Float32} (probability distribution)
#   - pi_mat  : M×(n*M) Matrix{Float32} (the pi(y|x) values)
#   - W       : M×(n*M) Matrix{Float32} (the channel W(y|x))
#
# Outputs:
#   - F0_val  : Float32, value of F0(ρ,Q)
#   - F0p_val : Float32, value of F0'(ρ,Q)
#   - F0pp_val: Float32, value of F0''(ρ,Q)
###############################################################################
function computeF0_all_metal(rho::Float32,
                             Q::Vector{Float32},
                             pi_mat::Matrix{Float32},
                             W::Matrix{Float32})

    A = rand(100)
    dA = MetalArray(A)   # create a GPU array on Metal
    
    # Convert to MetalArrays
    println("1. GPU Data Transfer:")
    @time begin
        Wd     = MetalArray(W)
        pid    = MetalArray(pi_mat)
    end

    println("\n2. Computing basic operations:")
    @time begin
        s   = 1f0 / (1f0 + rho)
        sp  = -1f0 / (1f0 + rho)^2
        spp =  2f0 / (1f0 + rho)^3
    end

    println("\n3. GPU Operations (log, exp):")
    @time begin
        logW = @. log(Wd)
        A    = @. exp(s * logW)
        D    = @. exp(-s*rho * logW)
        pi_term = @. pid * D
    end

    println("\n4. CPU Operations (matrix multiplications):")
    @time begin
        A_cpu = Array(A)
        B_cpu = (Q' * A_cpu)
        B     = B_cpu
        logB_cpu = log.(B_cpu)
        C_cpu = exp.(rho .* logB_cpu)
    end

    println("\n5. Computing F0:")
    @time begin
        pi_term_cpu = Array(pi_term)
        M1_cpu = Q' * pi_term_cpu
        F0_val = M1_cpu * C_cpu'
    end

    println("\n6. Computing F0':")
    @time begin
        partX_gpu = @. (sp * logW) * A
        partX     = Array(partX_gpu)
        X_cpu     = Q' * partX
        X_over_B  = X_cpu ./ B_cpu
        T1_cpu    = logB_cpu .+ (rho .* X_over_B)

        factorA_gpu = @. (-sp*rho - s) * logW
        T1mat_gpu   = pi_term .* factorA_gpu
        T1mat       = Array(T1mat_gpu)
        term1       = (Q' * T1mat) * C_cpu'

        term2       = M1_cpu * (C_cpu .* T1_cpu)'
        F0p_val     = term1 + term2
    end

    println("\n7. Computing F0'':")
    @time begin
        factorA2_gpu = @. factorA_gpu * factorA_gpu
        factorB_gpu  = @. (-spp*rho - 2f0*sp) * logW
        T1_sq_cpu    = T1_cpu .^ 2f0

        comb_gpu = @. (spp*logW + (sp*logW)*(sp*logW)) * A
        comb_cpu = Array(comb_gpu)
        X2_cpu   = Q' * comb_cpu
        X2_over_B = X2_cpu ./ B_cpu

        TAmat_gpu = pi_term .* factorA2_gpu
        TAmat     = Array(TAmat_gpu)
        termA     = (Q' * TAmat) * C_cpu'

        TBmat_gpu = pi_term .* factorB_gpu
        TBmat     = Array(TBmat_gpu)
        termB     = (Q' * TBmat) * C_cpu'

        factorA_mat_gpu = pi_term .* factorA_gpu
        factorA_mat     = Array(factorA_mat_gpu)
        termC           = 2f0 * (Q' * factorA_mat) * (C_cpu .* T1_cpu)'

        termD = M1_cpu * (C_cpu .* T1_sq_cpu)'

        extra_cpu = 2f0 .* X_over_B .+ (rho .* X2_over_B) .- (rho .* (X_over_B .^ 2f0))
        termE     = M1_cpu * (C_cpu .* extra_cpu)'

        F0pp_val  = termA + termB + termC + termD + termE
    end

    return F0_val, F0p_val, F0pp_val
end

function gauss_hermite(n::Int)
    # Main diagonal (alpha) of Hermite tridiagonal matrix is all zeros.
    a = zeros(n)
    # Off-diagonal (beta) values: sqrt(k/2) for k = 1..n-1
    b = [ sqrt(k/2) for k in 1:(n-1) ]
    
    # Form the symmetric tridiagonal matrix
    T = SymTridiagonal(a, b)
    
    # Compute its eigen-decomposition
    E = eigen(T)
    x = E.values   # Unsorted nodes
    V = E.vectors  # Corresponding eigenvectors
    
    # Weights: w[i] = (first-component of ith eigenvector)^2 * sqrt(pi)
    w = [ V[1, i]^2 * sqrt(pi) for i in 1:n ]
    
    # Sort by ascending node order
    p = sortperm(x)
    x_sorted = x[p]
    w_sorted = w[p]
    
    return x_sorted, w_sorted
end

function createPiMatrix(M, N, weights)
    # Initialize matrix with zeros
    pi_matrix = zeros(Float32, M, N^2 * M)
    
    # For each row
    for i in 1:M
        # For all combinations of weights
        idx = 1
        for j in 1:N
            for k in 1:N
                # Calculate weight product
                pi_block = weights[j] * weights[k]
                # Calculate column position
                col = (i - 1) * N^2 + idx
                # Assign the weight product
                pi_matrix[i, col] = pi_block
                idx += 1
            end
        end
    end
    
    return pi_matrix
end

function createGMatrix(X, z_matrix, SNR)
    # Number of source symbols
    M = length(X)
    
    # Assuming z_matrix is N-by-N
    N = size(z_matrix, 1)
    
    # Initialize the output matrix
    g_matrix = zeros(M, M * N^2)
    
    # Loop over "received symbols" (j)
    for j in 1:M
        # Precompute d for this j: sqrt(SNR) * (X - X[j])
        d = sqrt(SNR) .* (X .- X[j])  # elementwise .- and .* for broadcasting
        
        # Column indices for the current block (corresponding to j)
        col_idx = ((j - 1) * N^2 + 1) : (j * N^2)
        
        # Loop over "source symbols" (i)
        for i in 1:M
            # Compute G over all quadrature nodes:
            #   z_matrix is N×N, d[i] is scalar, so we do z_matrix .+ d[i].
            g_vals = Float32.((1f0 / π) .* exp.(-abs2.(z_matrix .+ d[i])));
            
            # g_vals = Float32((1 / π) * exp(-abs2(z_matrix .+ d[i])))
            # g_vals = G.(z_matrix .+ d[i])  # broadcast the function G
            
            # Flatten the NxN matrix into a vector, then store in row i
            g_matrix[i, col_idx] = vec(g_vals)' 
            # (vec(g_vals) is a 1D vector of length N^2; 
            #  the apostrophe ' performs a transpose => 1×(N^2) row)
        end
    end
    
    return g_matrix
end

function createComplexNodesMatrix(nodes)
    N = length(nodes)
    # Initialize an N×N array of complex zeros
    z_matrix = zeros(ComplexF64, N, N)
    
    for i in 1:N
        for j in 1:N
            z_matrix[i, j] = nodes[i] + im * nodes[j]
        end
    end
    
    return z_matrix
end

function generatePAMConstellation(M::Int, d::Real)
    # Create a range from -half to half in steps of 1, then scale by d
    half = (M - 1) / 2
    pam = collect(-half:1:half) .* d
    return Float32.(pam)  # Convert to Float32
end

###############################################################################
# Test / Demo
###############################################################################

function testF0Metal()
    Random.seed!(1234)

    println("Generating test data...")
    # Example dimensions
    M = 2
    n = 2
    SNR = Float32(1.0)  # Convert to Float32
    G(z) = Float32((1 / π) * exp(-abs2(z)))  # Ensure Float32 output

    # Generate PAM constellation and normalize power
    X = generatePAMConstellation(M, 2)
    X ./= sqrt(mean(X .^ 2))  # Normalize average power to 1

    # Create test data
    nodes, weights = gauss_hermite(n)
    Z_cpu = createComplexNodesMatrix(nodes)
    W_cpu = Float32.(createGMatrix(X, Z_cpu, SNR))  # Convert to Float32
    pi_cpu = Float32.(createPiMatrix(M, n, weights))   # Convert to Float32
    Q_cpu = fill(Float32(1/M), M)  # Create uniform distribution in Float32

    println("\nMatrices:")
    println("W_cpu: $W_cpu")
    println("pi_cpu: $pi_cpu")
    println("Q_cpu: $Q_cpu")

    rho_test = 0.5f0

    println("\nRunning computation...")
    # Time the main computation
    time_result = @timed computeF0_all_metal(rho_test, Q_cpu, pi_cpu, W_cpu)

    println("\nResults:")
    println("F0   = $(time_result.value[1])")
    println("F0'  = $(time_result.value[2])")
    println("F0'' = $(time_result.value[3])")
    println("\nPerformance:")
    println("Total computation time: $(round(time_result.time, digits=4)) seconds")
    println("Memory allocated: $(round(time_result.bytes/1024/1024, digits=2)) MB")
end

# Run the test
testF0Metal()