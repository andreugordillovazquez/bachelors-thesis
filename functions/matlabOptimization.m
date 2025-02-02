function matlabOptimization()
    % Initialize parameters
    params = initializeParameters();
    
    % Create matrices
    [pi_matrix, g_matrix] = createAllMatrices(params);
    
    % Set rate
    R = 0.5;
    
    % Create objective function (negative since fmincon minimizes)
    objective = @(rho) -computeER(rho, R, params.Q, pi_matrix, g_matrix);
    
    % Set optimization options
    options = optimoptions('fmincon', ...
        'Display', 'iter', ...        % Show iteration output
        'OptimalityTolerance', 1e-6, ...
        'PlotFcn', @optimplotfval);   % Plot progress
    
    % Set bounds and constraints
    rho0 = 0.5;          % Initial guess
    lb = 0;             % Lower bound
    ub = 1;             % Upper bound
    
    % Run optimization
    [optimal_rho, fval] = fmincon(objective, rho0, [], [], [], [], lb, ub, [], options);
    
    % Display results
    disp('================================================================');
    disp(' Optimization Results (using fmincon)');
    disp('----------------------------------------------------------------');
    disp(['Optimal rho: ', num2str(optimal_rho)]);
    disp(['Maximum E(R): ', num2str(-fval)]);  % Negative since we minimized -E(R)
    disp('================================================================');
end

function ER = computeER(rho, R, Q, pi_mat, G)
    % Compute E_0(rho)
    E_o = E0_of_rho(rho, Q, pi_mat, G);
    
    % Compute E(R)
    ER = E_o - rho * R;
end

function val = E0_of_rho(rho, Q, pi_mat, G)
    % Calculate intermediate terms using elementwise operations
    g_power_neg = G.^(-rho/(1+rho));  % First power term
    g_power_pos = G.^(1/(1+rho));     % Second power term
    
    % Calculate first part: Q * (pi_mat .* g_power_neg)
    first_part = Q * (pi_mat .* g_power_neg);
    
    % Calculate second part: (Q * g_power_pos).^rho
    second_part = (Q * g_power_pos).^rho;
    
    % Combine parts
    val = -log2((1/pi) * first_part * second_part');
end

function [pi_matrix, g_matrix] = createAllMatrices(params)
    % Create complex quadrature nodes matrix
    z_matrix = createComplexNodesMatrix(params.nodes);
    
    % Create Pi matrix
    pi_matrix = createPiMatrix(params.M, params.N, params.weights);
    
    % Create G matrix
    G = @(z) (1/pi) * exp(-abs(z).^2);
    g_matrix = createGMatrix(params.X, z_matrix, params.SNR, G);
end

function displayParameters(params)
    disp('================================================================');
    disp(' Gaussian-Hermite Quadrature');
    disp('----------------------------------------------------------------');
    disp(['Quadrature Nodes: ', mat2str(params.nodes, 4)]);
    disp(['Quadrature Weights: ', mat2str(params.weights, 4)]);
    disp(['Dimensions: ', num2str(params.N), ' nodes']);
    disp('----------------------------------------------------------------');
    disp(' Constellation Points');
    disp('----------------------------------------------------------------');
    disp(['Points: ', mat2str(params.X, 4)]);
    disp(['Dimensions: ', num2str(length(params.X)), ' points']);
    disp(['Average Energy: ', num2str(mean(params.X.^2), 4), ' (normalized)']);
    disp('----------------------------------------------------------------');
    disp(' Probability Distribution');
    disp('----------------------------------------------------------------');
    disp(['Distribution: ', mat2str(params.Q, 4)]);
    disp(['Dimensions: ', num2str(length(params.Q))]),
end