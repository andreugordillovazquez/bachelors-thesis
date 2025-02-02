function params = initializeParameters()
% initializeParameters Initializes the parameters for E0 computation
%
% Output:
% params - A structure containing the following fields:
%   M_const: Number of constellation points
%   SNR: Signal-to-Noise Ratio
%   num_points: Number of points for evaluating ρ
%   N: Number of quadrature nodes
%   nodes: Quadrature nodes (array of size [N])
%   weights: Quadrature weights (array of size [N])
%   X: Normalized constellation points (array of size [M_const])
%   Q: Probability distribution for constellation points (array of size [M_const])

    % Initialize basic parameters
    params.M = 2;      % Constellation size
    params.SNR = 1;          % Signal-to-Noise Ratio
    params.num_points = 1000; % Number of points for rho
    params.N = 2;            % Number of quadrature nodes

    % Generate Gaussian-Hermite quadrature nodes and weights
    [params.nodes, params.weights] = GaussHermite_Locations_Weights(params.N);

    % Display quadrature information with clear formatting
    disp('================================================================');
    disp('                 Gaussian-Hermite Quadrature');
    disp('----------------------------------------------------------------');
    disp(['Quadrature Nodes:   ', mat2str(params.nodes, 4)]);
    disp(['Quadrature Weights: ', mat2str(params.weights, 4)]);
    disp(['Dimensions:         ', num2str(params.N), ' nodes']);
    disp('----------------------------------------------------------------');

    % Generate constellation points
    X = zeros(1, params.M);
    aux = -(params.M - 1);
    for i = 1:params.M
        X(i) = aux;
        aux = aux + 2;
    end

    % Normalize constellation points
    avg_energy = mean(X.^2);
    params.X = X / sqrt(avg_energy);

    % Display constellation information
    disp('                    Constellation Points');
    disp('----------------------------------------------------------------');
    disp(['Points:             ', mat2str(params.X, 4)]);
    disp(['Dimensions:         ', num2str(length(params.X)), ' points']);
    disp(['Average Energy:     ', num2str(mean(params.X.^2), 4), ' (normalized)']);
    disp('----------------------------------------------------------------');

    % Generate and display probability distribution
    params.Q = ones(1, params.M) / params.M;
    disp('                  Probability Distribution');
    disp('----------------------------------------------------------------');
    disp(['Distribution:       ', mat2str(params.Q, 4)]);
    disp(['Dimensions:         ', num2str(length(params.Q)), ' probabilities']);
    disp(['Sum of Prob:        ', num2str(sum(params.Q), 4)]);
    disp('================================================================');
end