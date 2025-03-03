function [F0, dF0, d2F0] = derivativesE0(Q, pi_matrix, w_matrix, rho)
    % tic;
    % Precompute constants
    s   = 1/(1+rho);
    sp  = -1/(1+rho)^2;
    spp = 2/(1+rho)^3;
    
    % Compute log(W) once (for numerical stability)
    logW = log(w_matrix);
    
    % Compute the "pi_term": π .* W^{-s*rho}
    pi_term = pi_matrix .* exp(-s * rho * logW);
    
    % Compute inner exponentials: W^s and then (Q' * W^s)
    inner_exp   = exp(s * logW);     % same as W.^s
    Q_inner     = Q' * inner_exp;      % 1 x (n*M)
    log_Q_inner = log(Q_inner);
    outer_exp   = exp(rho * log_Q_inner);
    outer_term  = outer_exp.';         % Transpose to get column vector
    
    % initial = toc;

    % tic;
    % Final computation for F0 and E0
    F0 = Q' * pi_term * outer_term;
    % E0 = -log2(F0);
    % first = toc;
    
    %% Compute first derivative dF0
    % tic;
    % Precompute common factor for derivative (combine terms from s and sp)
    common_factor = sp * rho + s;  % so that term1 = -common_factor * logW
    term1_part = -common_factor * logW;
    
    % Term 1: derivative from pi_term component
    Term1 = Q' * (pi_term .* term1_part) * outer_term;
    
    % Term 2: derivative from the outer_term component
    inner_exp_logW = inner_exp .* logW; % reuse this term
    numerator = (Q' * inner_exp_logW) * sp;  % 1 x (n*M)
    d_log_Q_inner = numerator ./ Q_inner;      % elementwise division (1 x (n*M))
    derivative_outer_part = log_Q_inner + rho * d_log_Q_inner;
    d_outer_exp = outer_exp .* derivative_outer_part;
    d_outer_term = d_outer_exp.';  % column vector
    Term2 = Q' * pi_term * d_outer_term;
    
    % Total first derivative
    dF0 = Term1 + Term2;
    % dE0 = -dF0 / (F0 * log(2));  % using natural log for conversion
    % second = toc;
    
    %% Compute second derivative d2F0
    % tic;
    % Precompute parts from the first derivative that we need squared
    term1_sq = term1_part.^2;
    common_factor2 = spp * rho + 2 * sp;  % from derivative of (sp*rho+s)
    part2 = -common_factor2 * logW;  % equivalent to (-spp*rho - 2*sp)*logW
    
    % Term 1: from square of the first derivative term (pi_term component)
    Term1_2 = Q' * (pi_term .* term1_sq) * outer_term;
    
    % Term 2: from the second derivative of pi_term component
    Term2_2 = Q' * (pi_term .* part2) * outer_term;
    
    % Term 3: cross term from derivative of outer_term
    Term3_2 = 2 * Q' * (pi_term .* term1_part) * d_outer_term;
    
    % Term 4: second derivative from the outer_term (square of derivative factor)
    Term4_2 = Q' * pi_term * (outer_exp .* (derivative_outer_part.^2)).';
    
    % Term 5: additional second derivative contribution from outer_term
    % Component 1:
    numerator_component1 = Q' * inner_exp_logW;
    comp1 = 2 * sp * (numerator_component1 ./ Q_inner);
    % Component 2:
    logW_sq = logW.^2;
    combined = spp * logW + sp^2 * logW_sq;
    numerator_component2 = Q' * (inner_exp .* combined);
    comp2 = rho * (numerator_component2 ./ Q_inner);
    % Component 3:
    comp3 = -rho * ((sp * (numerator_component1 ./ Q_inner)).^2);
    
    Term5_2 = Q' * pi_term * (outer_exp .* (comp1 + comp2 + comp3)).';
    
    % Sum terms to get d2F0
    d2F0 = Term1_2 + Term2_2 + Term3_2 + Term4_2 + Term5_2;
    
    % Finally, compute the second derivative d2E0 from F0, dF0, and d2F0
    % d2E0 = -(d2F0 * F0 - dF0^2) / (F0^2 * log(2));
    % third = toc; 

    % times = [initial, first, second, third];
end