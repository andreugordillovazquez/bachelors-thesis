function [optimal_rho, max_value] = optimizeRho_fminbnd(Q, pi_matrix, g_matrix, R)
% optimizeRhoFixed - Optimized function to find ρ that maximizes E0(ρ) - ρR
%
% Inputs:
%   Q         - Probability distribution vector (1×M)
%   pi_matrix - Quadrature weights matrix (M×N²M)
%   g_matrix  - G function matrix (M×N²M)
%   R         - Communication rate R
%
% Outputs:
%   optimal_rho - The optimal ρ value in [0,1]
%   max_value   - The maximum value of E0(ρ) - ρR

    % Define objective function combining E0(ρ) computation and rate term
    % Using nested function for better performance and cleaner scope
    function neg_value = objective(rho)
        % Direct computation of E0(ρ) - ρR
        rho_factor = 1 / (1 + rho);
        g_powered = exp(rho_factor * log(g_matrix));
        qg_rho = Q * g_powered;
        pi_g_rho = pi_matrix .* exp(-rho * rho_factor * log(g_matrix));
        Eo_rho = -log2((1/pi) * (Q * (pi_g_rho * (qg_rho.^rho)')));
        
        % Return negative for minimization
        neg_value = -(Eo_rho - R * rho);
    end

    % Configure optimization options
    options = optimset('TolX', 1e-6, ...
                      'Display', 'iter', ...
                      'MaxIter', 100, ...
                      'MaxFunEvals', 100);

    % Perform optimization
    [optimal_rho, max_neg_value] = fminbnd(@objective, 0, 1, options)
    
    % Convert result back to maximum value
    max_value = -max_neg_value;
end
