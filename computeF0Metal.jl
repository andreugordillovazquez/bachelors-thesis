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

    println("1. GPU Data Transfer:")
    @time begin
        Wd     = MtlArray(W)
        pid    = MtlArray(pi_mat)
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

###############################################################################
# Test / Demo
###############################################################################

function testF0Metal()
    Random.seed!(1234)

    println("Generating test data...")
    # Example dimensions
    M = 1024
    n = 20

    # Create random W, pi, Q in Float32
    W_cpu       = rand(Float32, M, M*n) .+ 0.1f0  # offset to avoid log(0)
    pi_cpu      = rand(Float32, M, M*n)
    Q_cpu       = rand(Float32, M)
    rho_test    = 0.5f0

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