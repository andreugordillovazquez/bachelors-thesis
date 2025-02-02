function [rho_opt, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star)
%   Optimize the E0 function using Newton's method.
%
%   [rho_opt, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star)
%
%   This function applies Newton's method to find the optimal value of the parameter
%   rho that maximizes the function E0, whose value and derivatives are computed by the
%   function 'derivativesE0'. The iteration updates rho using the gradient and curvature
%   of E0 until the change in rho is smaller than the specified tolerance or a maximum
%   number of iterations is reached.
%
%   Inputs:
%       Q         - Matrix or data structure required for computing E0.
%       pi_matrix - Matrix of probability values or related parameters for the E0 function.
%       g_matrix  - Matrix containing gain values or similar parameters used in E0's computation.
%       R         - Scalar parameter representing a rate or threshold used in the optimization.
%       tol       - Tolerance for the convergence criterion based on the change in rho.
%       rho_star  - Initial guess for the value of rho.
%
%   Outputs:
%       rho_opt   - The optimized value of rho at which E0 is maximized.
%       E0_max    - The value of the function E0 evaluated at rho_opt.
%

    iter = uint32(0);
    log2_const = log(2);
    rho = rho_star;
    max_iter = 100;
    tol_curvature = 1e-10;
    
    while (iter < max_iter)
        % Compute all derivatives in one pass to avoid redundant calculations
        [F0, E0, dF0, ~, d2F0, ~] = derivativesE0(Q, pi_matrix, g_matrix, rho);

        % Optimize divisions by precomputing repeated values
        f_inv = 1/F0;
        fp_f = dF0 * f_inv;

        % Compute gradient and curvature with minimized operations
        gradient = (-fp_f / log2_const) - R;
        curvature = -((d2F0 * f_inv) - (fp_f * fp_f)) / log2_const;
        
        % Combine convergence checks to reduce branching
        if abs(gradient/curvature) < tol || abs(curvature) < tol_curvature
            break;
        end
        
        disp(rho);

        % Compute new rho value and clamp to [0, 1]
        rho_new = rho - gradient / curvature;
    
        % Check convergence based on rho change
        if abs(rho_new - rho) < tol
            break;
        end
    
        rho = rho_new;
        iter = iter + 1;
    end

    % Final output:
    rho_opt = rho;
    % E0(rho_opt) (the function value at final rho):
    E0_max = E0;

    fprintf('\nNewton algorithm finished after %u iterations.\n', iter);
end
