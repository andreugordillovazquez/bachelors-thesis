% computeFirstDerivativeE0
% Computes the first derivative of the Gallager E0 function for a given channel.
%
% Inputs:
%   Q         - Input distribution vector
%   pi_matrix - Matrix of input probabilities
%   w_matrix  - Channel transition probability matrix
%   rho       - Exponent parameter (scalar)
%
% Output:
%   dE0       - First derivative of E0 with respect to rho

function dE0 = computeFirstDerivativeE0(Q, pi_matrix, w_matrix, rho)
    s = 1/(1+rho);           % s parameter
    sp = -1/(1+rho)^2;       % Derivative of s with respect to rho

    log_W = log(w_matrix);   % Logarithm of channel matrix for stability
    
    pi_term = pi_matrix .* exp(-s * rho * log_W); % π .* W^{-sρ}
    
    inner_exp = exp(s * log_W);                   % W^s
    Q_inner = Q' * inner_exp;                     % Q^T * W^s
    
    log_Q_inner = log(Q_inner);                   % Log for outer exponent
    outer_exp = exp(rho * log_Q_inner);           % (Q^T * W^s)^ρ
    outer_term = outer_exp.';                     % Transpose to column
    
    F0 = Q' * pi_term * outer_term;               % Main function value

    % First derivative terms
    term1_part = (-sp * rho * log_W) - (s * log_W);
    Term1 = Q' * (pi_term .* term1_part) * outer_term;
    
    inner_exp_logW = inner_exp .* log_W;
    derivative_outer_part = log_Q_inner + rho * ((Q' * inner_exp_logW * sp) ./ Q_inner);
    Term2 = Q' * pi_term * (outer_exp .* derivative_outer_part).';
    
    F0_prime = Term1 + Term2;                     % Derivative of F0
    dE0 = -F0_prime / (F0 * log(2));              % Final derivative
end
