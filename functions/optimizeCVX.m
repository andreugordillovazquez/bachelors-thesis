function [optimal_rho, max_value] = optimizeCVX(Q, pi_matrix, g_matrix, R)

% CVX optimization for maximizing E0(ρ) - ρR

% Ensure CVX is set up and initialized
cvx_begin
    % Declare rho as a scalar variable between 0 and 1
    variable rho
    
    % Define the objective function
    % Note: CVX minimizes by default, so we negate to maximize
    Eo = computeEoForRhoExponential(rho, Q, pi_matrix, g_matrix);
    objective = -(Eo - R * rho);
    
    % Set optimization constraints
    subject to
        0 <= rho <= 1;
    
    % Minimize the negative objective (equivalent to maximizing)
    minimize(objective)
cvx_end

% Display results
optimal_rho = rho;
max_value = -cvx_optval;

disp(['Optimal ρ: ', num2str(optimal_rho)]);
disp(['Maximum value: ', num2str(max_value)]);



end