function [rho_opt, E0_max] = optimizationGradientDescent(Q, pi_matrix, g_matrix, R, tol, rho_star)
%OPTIMIZATIONGRADIENTDESCENT Optimizes G(rho) = E0(rho) - rho*R using gradient ascent
%
%   [rho_opt, E0_max] = optimizationGradientDescent(Q, pi_matrix, g_matrix, R, tol, rho_star)
%
%   This function uses gradient ascent with an adaptive step size computed as
%       alpha = -1/curvature,
%   where curvature is computed from the second derivative of E0.
%
%   Inputs:
%       Q         - Data vector required by derivativesE0.
%       pi_matrix - Matrix of quadrature weight products.
%       g_matrix  - Matrix of G function values.
%       R         - Scalar rate parameter.
%       tol       - Tolerance for convergence (|rho_new - rho|).
%       rho_star  - Initial guess for rho (should be within [0,1]).
%
%   Outputs:
%       rho_opt   - Optimized value of rho (within [0,1]).
%       E0_max    - Value of E0 evaluated at rho_opt.
%

    iter = 1;
    max_iter = 100;       % Maximum iterations for the loop
    rho = rho_star;       % current value of rho
    log2_const = log(2);
    
    while iter < max_iter
        % Evaluate derivatives at current rho.
        [F0, ~, dF0, ~, d2F0, ~] = derivativesE0(Q, pi_matrix, g_matrix, rho);
        
        % Check for a valid F0 to avoid division by zero.
        if F0 <= 0 || isnan(F0)
            warning('F0 is non-positive or NaN at rho = %f. Terminating optimization.', rho);
            break;
        end
        
        % Precompute common factors.
        f_inv = 1 / F0;
        fp_f  = dF0 * f_inv;
        
        % Compute gradient: G'(rho)=E0'(rho)-R, where E0'(rho) = - (dF0/(F0*ln2))
        gradient = (-fp_f / log2_const) - R;
        
        % Compute curvature: G''(rho)= -((d2F0/F0) - (dF0/F0)^2)/ln2.
        curvature = -((d2F0 * f_inv) - (fp_f * fp_f)) / log2_const;
        
        % Use fallback if curvature is too close to zero.
        if isnan(curvature) || abs(curvature) < 1e-12
            warning('Curvature is near zero at rho = %f. Using fallback step size.', rho);
            alpha = 1e-3;
        else
            % For maximization Newton update, define:
            % alpha = -1/curvature, so that:
            % rho_new = rho + alpha * gradient = rho - gradient/curvature.
            alpha = -1 / curvature;
        end
        
        % Update rho using the adaptive step size.
        rho_new = rho + alpha * gradient;
        
        % Keep rho within [0,1]
        rho_new = min(max(rho_new, 0), 1);
        
        % Display intermediate values for debugging:
        fprintf('iter = %d, rho = %.6f, F0 = %.6e, dF0 = %.6e, d2F0 = %.6e\n', ...
            iter, rho, F0, dF0, d2F0);
        fprintf('gradient = %.6e, curvature = %.6e, alpha = %.6e\n', gradient, curvature, alpha);
        
        % Check for convergence.
        if abs(rho_new - rho) < tol
            rho = rho_new;
            break;
        end
        
        % Update rho and iteration count.
        rho = rho_new;
        iter = iter + 1;
    end

    % Final result.
    rho_opt = rho;
    [~, E0_opt, ~, ~, ~, ~] = derivativesE0(Q, pi_matrix, g_matrix, rho_opt);
    E0_max = E0_opt;

    fprintf('\nGradient Ascent finished after %d iterations.\n', iter);
    fprintf('rho_opt = %.6f\n', rho_opt);
    fprintf('E0_max = %.6f\n\n', E0_max);
end
