function Eo_rho = computeEoForRho(rho, Q, pi_matrix, g_matrix)
% computeEoForRho - Computes E0(ρ) for a given ρ value in the optimization of 
% error exponent E(R) = max{E0(ρ) - ρR} for communication systems
%
% Inputs:
%   rho       - Value of ρ parameter, 0 ≤ ρ ≤ 1
%   Q         - Probability distribution of constellation points (1×M)
%   pi_matrix - Matrix of quadrature weights products (M×N²M)
%   g_matrix  - Matrix of G function values for each combination (M×N²M)
%
% Output:
%   Eo_rho    - Computed value of E0(ρ) for the given parameters

    % Compute the intermediate terms
    g_powered = g_matrix.^(1 / (1 + rho));
    qg_rho = Q' * g_powered;     % Weighted sum over g_powered
    qg_rho_t = qg_rho.^rho;     % Raise to the power of rho
    
    % Compute the Pi-weighted terms
    pi_g_rho = pi_matrix .* (g_matrix.^(-rho / (1 + rho)));
    
    % Combine results to compute the second component
    component_second = Q' * (pi_g_rho * qg_rho_t');
    
    % Compute the final value for E0(rho)
    Eo_rho = -log2((1 / pi) * component_second);
end