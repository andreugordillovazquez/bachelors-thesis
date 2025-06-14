% derivativesE0
% Computes F0, its first derivative dF0, and second derivative d2F0 with respect to rho.
%
% Inputs:
%   Q         - Input distribution vector
%   pi_matrix - Matrix of probability weights
%   w_matrix  - Channel or function value matrix
%   rho       - Exponent parameter
%
% Outputs:
%   F0   - Function value at rho
%   dF0  - First derivative with respect to rho
%   d2F0 - Second derivative with respect to rho

function [F0, dF0, d2F0] = derivativesE0(Q, pi_matrix, w_matrix, rho)
    s   = 1/(1+rho);
    sp  = -1/(1+rho)^2;
    spp = 2/(1+rho)^3;
    
    logW = log(w_matrix);                                 % Logarithm for stability
    pi_term = pi_matrix .* exp(-s * rho * logW);          % π .* W^{-s*rho}
    inner_exp   = exp(s * logW);                          % W^s
    Q_inner     = Q' * inner_exp;                         % Q^T * W^s
    log_Q_inner = log(Q_inner);
    outer_exp   = exp(rho * log_Q_inner);                 % (Q^T * W^s)^rho
    outer_term  = outer_exp.';                            % Column vector

    F0 = Q' * pi_term * outer_term;                       % Function value

    % First derivative
    common_factor = sp * rho + s;
    term1_part = -common_factor * logW;
    Term1 = Q' * (pi_term .* term1_part) * outer_term;
    inner_exp_logW = inner_exp .* logW;
    numerator = (Q' * inner_exp_logW) * sp;
    d_log_Q_inner = numerator ./ Q_inner;
    derivative_outer_part = log_Q_inner + rho * d_log_Q_inner;
    d_outer_exp = outer_exp .* derivative_outer_part;
    d_outer_term = d_outer_exp.';
    Term2 = Q' * pi_term * d_outer_term;
    dF0 = Term1 + Term2;

    % Second derivative
    term1_sq = term1_part.^2;
    common_factor2 = spp * rho + 2 * sp;
    part2 = -common_factor2 * logW;
    Term1_2 = Q' * (pi_term .* term1_sq) * outer_term;
    Term2_2 = Q' * (pi_term .* part2) * outer_term;
    Term3_2 = 2 * Q' * (pi_term .* term1_part) * d_outer_term;
    Term4_2 = Q' * pi_term * (outer_exp .* (derivative_outer_part.^2)).';
    numerator_component1 = Q' * inner_exp_logW;
    comp1 = 2 * sp * (numerator_component1 ./ Q_inner);
    logW_sq = logW.^2;
    combined = spp * logW + sp^2 * logW_sq;
    numerator_component2 = Q' * (inner_exp .* combined);
    comp2 = rho * (numerator_component2 ./ Q_inner);
    comp3 = -rho * ((sp * (numerator_component1 ./ Q_inner)).^2);
    Term5_2 = Q' * pi_term * (outer_exp .* (comp1 + comp2 + comp3)).';
    d2F0 = Term1_2 + Term2_2 + Term3_2 + Term4_2 + Term5_2;
end
