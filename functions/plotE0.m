function plotE0(rho_values, Eo_rho, params)
% plotE0 Visualizes how E0(ρ) changes as we vary rho
%
% Inputs:
%   rho_values: Range of ρ values from 0 to 1 
%   Eo_rho: Computed E0 values corresponding to each ρ
%   params: Structure containing system parameters, including:
%          - nodes: Quadrature nodes used in computation
%
% The resulting plot shows:
%   x-axis: ρ values from 0 to 1
%   y-axis: E0(ρ) values

   close all;
   
   % Create a new figure window
   figure;
   
   % Plot E0(ρ)
   plot(rho_values, Eo_rho, 'linewidth', 2);
   
   % Label the plot
   title(['E_0(ρ) using ', num2str(length(params.nodes)), ' quadrature nodes']);
   
   % Label axes and add a grid
   xlabel('ρ');
   ylabel('E_0(ρ)');
   grid on;
   
   % Optional: Print key points about the curve
   disp('================================================================');
   disp('                       E0(ρ) Plot Analysis');
   disp('----------------------------------------------------------------');
   disp(['Maximum E0 value: ', num2str(max(Eo_rho), '%.4f')]);
   disp(['E0 at ρ = 0: ', num2str(Eo_rho(1), '%.4f')]);
   disp(['E0 at ρ = 1: ', num2str(Eo_rho(end), '%.4f')]);
   disp('================================================================');
end