function dE0 = computeFirstDerivativeE0(Q, pi_matrix, w_matrix, rho)
    s = 1/(1+rho);
    sp = -1/(1+rho)^2;

    % Compute log(W) for numerical stability
    log_W = log(w_matrix);
    
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

    % First derivative components
    term1_part = (-sp * rho * log_W) - (s * log_W);
    Term1 = Q' * (pi_term .* term1_part) * outer_term;
    
    inner_exp_logW = inner_exp .* log_W;
    derivative_outer_part = log_Q_inner + rho * ((Q' * inner_exp_logW * sp) ./ Q_inner);
    Term2 = Q' * pi_term * (outer_exp .* derivative_outer_part).';
    
    % Compute derivative
    F0_prime = Term1 + Term2;
    dE0 = -F0_prime / (F0 * log(2));
end

