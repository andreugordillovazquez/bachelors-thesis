% hermiteInterpolation
% Approximates the maximizing rho for G(rho) = E0(rho) - rho*R using cubic Hermite interpolation,
% given values and derivatives at rho = 0 and rho = 1. Also plots the interpolant and candidates.
%
% Inputs:
%   E0_0   - E0(0), value at rho = 0
%   E0p_0  - E0'(0), derivative at rho = 0
%   E0_1   - E0(1), value at rho = 1
%   E0p_1  - E0'(1), derivative at rho = 1
%   R      - Rate parameter
%
% Output:
%   rho_star - Value of rho in [0,1] maximizing the Hermite interpolant

function rho_star = hermiteInterpolation(E0_0, E0p_0, E0_1, E0p_1, R)
    % Set endpoint and derivative values for G(rho)
    A = E0_0;
    B = E0p_0 - R;
    C = E0_1 - R;
    D = E0p_1 - R;

    % Build cubic Hermite interpolant for G(rho)
    [p, ~, coefs] = cubicHermiteFromEndpoints(A, B, C, D);

    % Find real roots of the derivative in [0,1]
    roots_dp = roots([3*coefs(4) 2*coefs(3) coefs(2)]);
    roots_dp = roots_dp(imag(roots_dp)==0 & roots_dp>=0 & roots_dp<=1);

    % Evaluate p at endpoints and critical points, select maximizing rho
    candidates = [0; 1; roots_dp(:)];
    [~, idx] = max(p(candidates));
    rho_star = candidates(idx);

    % ---- Plotting ----
    rho_plot = linspace(0, 1, 200);
    G_plot = p(rho_plot);
    G_candidates = p(candidates);

    figure;
    plot(rho_plot, G_plot, 'b-', 'LineWidth', 1.5); hold on;
    plot(candidates, G_candidates, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    title('Cubic Hermite Interpolation for  G(\\rho) = E_0(\\rho) - \\rho R');
    xlabel('\\rho');
    ylabel('G(\\rho) = E_0(\\rho) - \\rho R');
    legend('G(\\rho) interpolant', 'Candidates (critical/endpoints)', 'Location', 'Best');
    grid on;
end

% cubicHermiteFromEndpoints
% Constructs cubic Hermite interpolant and its derivative for two points.
%
% Inputs:
%   A - Value at 0
%   B - Derivative at 0
%   C - Value at 1
%   D - Derivative at 1
%
% Outputs:
%   p     - Function handle for the cubic polynomial
%   dp    - Function handle for its derivative
%   coefs - Coefficient vector [a, b, c, d]

function [p, dp, coefs] = cubicHermiteFromEndpoints(A, B, C, D)
    a = A;
    b = B;
    c = 3*(C - A) - 2*B - D;
    d = -2*(C - A) + B + D;
    coefs = [a, b, c, d];

    p  = @(rho) a + b*rho + c*rho.^2 + d*rho.^3;
    dp = @(rho) b + 2*c*rho + 3*d*rho.^2;
end
