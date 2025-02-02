function performanceTest(params)
% runConstellationPerformanceTest Evaluates how computation time scales with constellation size
%
% Input:
%   params: Base configuration structure containing quadrature settings
%
% The results show both numerical timing data and a plot to help visualize
% how computation time grows with constellation size.

   % Define constellation sizes to test
   constellation_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028];
   execution_times = zeros(size(constellation_sizes));
   constellation_points = cell(size(constellation_sizes));
   
   % Test each constellation size
   for idx = 1:length(constellation_sizes)
       params.M_const = constellation_sizes(idx);
       
       % Generate normalized constellation points
       params.X = linspace(-3, 3, params.M_const);
       avg_energy = mean(params.X.^2);
       params.X = params.X / sqrt(avg_energy);
       constellation_points{idx} = params.X;
       
       % Set equal probabilities for all points
       params.Q = ones(1, params.M_const) / params.M_const;
       
       % Time the E0 computation
       tic;
       % Create necessary matrices
       z_matrix = createComplexNodesMatrix(params.nodes);
       pi_matrix = createPiMatrix(params.M_const, length(params.nodes), params.weights);
       g_matrix = createGMatrix(params.X, z_matrix, params.SNR, @(z) (1 / pi) * exp(-abs(z).^2));
       
       % Compute E0 for all ρ values
       rho_values = linspace(0, 1, params.num_points);
       Eo_rho = zeros(size(rho_values));
       for i = 1:length(rho_values)
           Eo_rho(i) = computeEoForRho(rho_values(i), params.Q, pi_matrix, g_matrix);
       end
       execution_times(idx) = toc;
       
       % Display progress with timing information
       disp(['M = ', num2str(params.M_const), ' points:']);
       disp(['    Execution time: ', num2str(execution_times(idx), '%.3f'), ' seconds']);
   end
   
   % Display summary statistics
   disp('----------------------------------------------------------------');
   disp('Performance Summary:');
   disp(['Fastest run: M = ', num2str(constellation_sizes(1)), ...
         ' points (', num2str(execution_times(1), '%.3f'), ' s)']);
   disp(['Slowest run: M = ', num2str(constellation_sizes(end)), ...
         ' points (', num2str(execution_times(end), '%.3f'), ' s)']);
   disp('================================================================');
   
   % At the end of your runConstellationPerformanceTest function, replace the plotting section with:

    % Create performance visualizations
    close all;  % Ensure clean plotting
    
    % Create a figure with two subplots side by side
    figure('Position', [100, 100, 1200, 500]);  % Make figure wider for side-by-side plots
    
    % Linear scale plot (left subplot)
    subplot(1, 2, 1);
    plot(constellation_sizes, execution_times, 'LineWidth', 2, 'Marker', 'o', ...
         'MarkerFaceColor', 'auto', 'MarkerSize', 8);
    title('Linear Scale: Computation Time vs. Constellation Size');
    xlabel('Constellation Size (M points)');
    ylabel('Computation Time (seconds)');
    grid on;
    
    % Logarithmic scale plot (right subplot)
    subplot(1, 2, 2);
    loglog(constellation_sizes, execution_times, 'LineWidth', 2, 'Marker', 'o', ...
           'MarkerFaceColor', 'auto', 'MarkerSize', 8);
    title('Log Scale: Computation Time vs. Constellation Size');
    xlabel('Constellation Size (M points)');
    ylabel('Computation Time (seconds)');
    grid on;
    
    % Add overall title
    sgtitle('E_0 Computation Performance Analysis', 'FontSize', 14);
    
    % Optional: Compute and display the approximate complexity
    p = polyfit(log10(constellation_sizes), log10(execution_times), 1);
    complexity_order = p(1);
    disp(['Approximate computational complexity: O(n^', num2str(complexity_order, '%.2f'), ')']);
end