function E0_rho = computeEoForRhoExponential(rho, Q, pi_matrix, g_matrix)
% computeEoForRho - Optimized computation of E0(ρ) for error exponent E(R)
%
% Inputs:
%   rho       - Value of ρ parameter, 0 ≤ ρ ≤ 1
%   Q         - Probability distribution of constellation points (1×M)
%   pi_matrix - Matrix of quadrature weights products (M×N²M)
%   g_matrix  - Matrix of G function values for each combination (M×N²M)
%
% Output:
%   Eo_rho    - Computed value of E0(ρ) for the given parameters

    % Precomputed commonly used terms
    s = 1/(1+rho);
    
    % Compute log(W) for numerical stability
    log_W = log(g_matrix);
    
    % Compute π .* W^{-sρ} using exponential and logarithm
    pi_term = pi_matrix .* exp(-s * rho * log_W);
    
    % Compute Q^T * W^s using logarithmic approach
    inner_exp = exp(s * log_W);
    Q_inner = Q' * inner_exp;
    
    % Compute (Q^T * W^s)^ρ and transpose to column vector
    log_Q_inner = log(Q_inner);
    outer_exp = exp(rho * log_Q_inner);
    outer_term = outer_exp.';
    
    % Final computation of F0
    F0 = Q' * pi_term * outer_term;
    E0_rho = -log2((1/pi)*F0);
end