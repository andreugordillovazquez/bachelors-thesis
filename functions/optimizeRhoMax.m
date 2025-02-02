function [optimal_rho, max_value] = optimizeRhoMax(Eo_rho, rho_values, R)
% optimizeRho Finds the optimal ρ that maximizes E0(ρ) - ρR for a given rate R
%
% Inputs:
%   Eo_rho: Vector of E0 values computed for different ρ
%   rho_values: Corresponding ρ values where E0 was evaluated
%   R: Value of the Communication Rate
%
% The function produces both numerical results and a visualization showing:
%   - The original E0(ρ) curve
%   - The optimization target E0(ρ) - ρR
%   - The optimal operating point

% disp('================================================================');
% disp('                    Optimization Parameters');
% disp('----------------------------------------------------------------');
% disp(['Rate (R): ', num2str(R), ' bits/channel use']);

% Calculate optimization target: maximize E0(ρ) - ρR
toOptimize = Eo_rho - R * rho_values;

% Find optimal point
[max_value, max_idx] = max(toOptimize);
optimal_rho = rho_values(max_idx);

% % Display optimization results
% % disp('----------------------------------------------------------------');
disp('Optimization Results:');
disp(['Optimal ρ: ', num2str(optimal_rho, '%.4f')]);
disp(['Maximum value: ', num2str(max_value, '%.4f')]);
disp(['E0 at optimal ρ: ', num2str(Eo_rho(max_idx), '%.4f')]);
% % disp('================================================================');

% close all;
% 
% % Plot both curves
% plot(rho_values, Eo_rho, 'LineWidth', 2, 'DisplayName', 'E_0(\rho)');
% hold on;
% plot(rho_values, toOptimize, 'LineWidth', 2, ...
%     'DisplayName', 'E_0(\rho) - \rho R');
% 
% % Mark optimal point
% plot(optimal_rho, max_value, 'ko', 'MarkerFaceColor', 'r', ...
%     'MarkerSize', 10, 'DisplayName', 'Optimal Point');
% 
% % Add vertical line at optimal rho
% plot([optimal_rho optimal_rho], [min(toOptimize) max_value], '--k', ...
%     'DisplayName', 'Optimal \rho');
% 
% xlabel('\rho', 'FontSize', 12);
% ylabel('Value', 'FontSize', 12);
% title({['Optimization Results for R = ', num2str(R)], ...
%     ['\rho_{opt} = ', num2str(optimal_rho, '%.4f'), ...
%     ', Maximum Value = ', num2str(max_value, '%.4f')]}, ...
%     'FontSize', 14);
% grid on;
% legend('Location', 'best');
% hold off;
end