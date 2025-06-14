% computeEoForRhoExponential
% Computes the Gallager E0(ρ) function for the error exponent E(R).
%
% Inputs:
%   rho       - Exponent parameter, 0 ≤ ρ ≤ 1 (scalar)
%   Q         - Probability distribution of constellation points (1×M)
%   pi_matrix - Matrix of quadrature weights products (M×N²M)
%   g_matrix  - Matrix of G function values for each combination (M×N²M)
%
% Output:
%   E0_rho    - Computed value of E0(ρ) for the given parameters

function E0_rho = computeEoForRhoExponential(rho, Q, pi_matrix, g_matrix)
    s = 1/(1+rho);                    % s parameter

    log_W = log(g_matrix);            % Logarithm of G matrix for stability

    pi_term = pi_matrix .* exp(-s * rho * log_W); % π .* W^{-sρ}

    inner_exp = exp(s * log_W);       % W^s
    Q_inner = Q' * inner_exp;         % Q^T * W^s

    log_Q_inner = log(Q_inner);       % Log for outer exponent
    outer_exp = exp(rho * log_Q_inner); % (Q^T * W^s)^ρ
    outer_term = outer_exp.';         % Transpose to column

    F0 = Q' * pi_term * outer_term;   % Main function value
    E0_rho = -log2((1/pi)*F0);        % Final E0(ρ) computation
end
