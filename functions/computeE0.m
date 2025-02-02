function [rho_values, Eo_rho] = computeE0(params)
% computeE0 Calculates E0(ρ) for a range of ρ values using Gaussian-Hermite quadrature
%
% Inputs:
%   params - Structure containing:
%     num_points: Number of points for ρ evaluation
%     nodes: Gaussian-Hermite quadrature nodes
%     weights: Gaussian-Hermite quadrature weights
%     X: Normalized constellation points
%     SNR: Signal-to-Noise Ratio
%     Q: Probability distribution for constellation points
%
% Outputs:
%   rho_values: Vector of ρ values from 0 to 1
%   Eo_rho: Corresponding E0 values for each ρ
%

   % Initialize output vectors
   % disp('================================================================');
   % disp('                    Computing E0(ρ) Function');
   % disp('----------------------------------------------------------------');
   
   % Generate ρ values from 0 to 1
   rho_values = linspace(0, 1, params.num_points);
   % rho_values = 0.5;
   Eo_rho = zeros(size(rho_values));
   
   % Create necessary matrices for computation
   % disp('Initializing computation matrices');
   z_matrix = createComplexNodesMatrix(params.nodes);
   pi_matrix = createPiMatrix(length(params.X), length(params.nodes), params.weights);
   
   % Define the Gaussian PDF function G(z)
   G = @(z) (1 / pi) * exp(-abs(z).^2);
   
   % Create G matrix for all constellation points
   g_matrix = createGMatrix(params.X, z_matrix, params.SNR, G);
   
   % Compute E0(ρ) for each ρ value
   % disp(['Computing E0 for ', num2str(params.num_points), ' points']);
   
   for idx = 1:length(rho_values)
       rho = rho_values(idx);
       Eo_rho(idx) = computeEoForRho(rho, params.Q, pi_matrix, g_matrix);
   end
   
   % Display E0 vector
   % disp('----------------------------------------------------------------');
   % disp('E0(ρ) values:');
   % disp(mat2str(Eo_rho, 4));
   % disp('================================================================');
end