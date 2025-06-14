% optimizationNewton
% Optimizes the E0 function using Newton's method.
%
% Inputs:
%   Q         - Input distribution or related data
%   pi_matrix - Matrix of probability weights
%   g_matrix  - Matrix of function values
%   R         - Rate parameter
%   tol       - Convergence tolerance for rho
%   rho_star  - Initial guess for rho
%
% Outputs:
%   rho_opt   - Optimized value of rho
%   E0_max    - Maximum value of E0 at rho_opt

function [rho_opt, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star)
    iter = 1;
    log2_const = log(2);
    rho = rho_star;
    max_iter = 100;
    tol_curvature = 1e-6;
    
    while (iter < max_iter)
        [F0, dF0, d2F0] = derivativesE0(Q, pi_matrix, g_matrix, rho); % Compute E0 and derivatives
        E0 = -log2((1/pi)*F0);                                        % E0 value

        f_inv = 1/F0;
        fp_f = dF0 * f_inv;

        gradient = (-fp_f / log2_const) - R;                          % Gradient
        curvature = -((d2F0 * f_inv) - (fp_f * fp_f)) / log2_const;   % Curvature
        
        if abs(gradient/curvature) < tol || abs(curvature) < tol_curvature
            break;
        end

        rho_new = rho - gradient / curvature;                         % Newton update
        if abs(rho_new - rho) < tol
            break;
        end

        rho = rho_new;
        iter = iter + 1;
    end

    rho_opt = rho;                                                    % Optimized rho
    E0_max = E0;                                                      % E0 at rho_opt

    fprintf('\nNewton algorithm finished after %u iterations.\n', iter);
end
