function rho_star = demoHermiteInterpolation(E0_0, E0p_0, E0_1, E0p_1, R)
    % Demo of constructing G(rho) = E0(rho) - rho*R
    % with known values/derivatives at rho=0 and rho=1,
    % then doing Hermite interpolation to approximate the maximum.
    %
    % It is assumed here that E0_values contains the exact (or independently
    % computed) values of G(rho) at a set of rho-points so that we can compare
    % with the interpolation result.

    % --- 1) Set your known endpoints and derivative values
    % For G(rho) = E0(rho) - rho*R we have:
    % G(0)  = E0(0)
    % G(1)  = E0(1) - R
    % G'(0) = E0'(0) - R
    % G'(1) = E0'(1) - R
    A = E0_0;
    B = E0p_0 - R;
    C = E0_1 - R;
    D = E0p_1 - R;

    % --- 2) Build the cubic Hermite interpolant for G(rho)
    % The helper function now also returns the coefficients.
    [p, ~, coefs] = hermiteInterp(A, B, C, D);

    % --- 3) Find critical points in [0,1]
    % The derivative of p(rho) is: dp(rho) = b + 2*c*rho + 3*d*rho^2,
    % where coefs = [a, b, c, d]. Compute the roots of:
    % 3*d*rho^2 + 2*c*rho + b = 0.
    roots_dp = roots([3*coefs(4) 2*coefs(3) coefs(2)]);
    roots_dp = roots_dp(imag(roots_dp)==0 & roots_dp>=0 & roots_dp<=1);  % real in [0,1]

    % Evaluate p at endpoints and those roots
    candidates = [0; 1; roots_dp];
    p_vals     = p(candidates);
    [p_max, idx] = max(p_vals);
    rho_star   = candidates(idx);

    % Print the approximate maximum G(rho_star)
    fprintf('Approx. max of G(rho) is %.4f at rho = %.4f\n', p_max, rho_star);

    % --- 4) Plot the interpolation and the provided "exact" values
    % figure; hold on; grid on;
    % rho_plot = linspace(0, 1, 200);
    % plot(rho_plot, p(rho_plot), 'b-', 'LineWidth', 1.5);
    % plot(candidates, p_vals, 'ro', 'MarkerSize', 8);
    % 
    % Here we assume that the vector E0_values represents the ρ-points (or the
    % exact G(ρ) values) computed externally. In this example we treat them as the
    % ρ-points and plot the interpolation evaluated at those points.
    % plot(E0_values, p(E0_values), 'g--', 'LineWidth', 1.5);
    % 
    % xlabel('\rho'); ylabel('G(\rho) = E_0(\rho) - \rho R');
    % title('Hermite Interpolation for G(\rho) = E_0(\rho) - \rho R');
    % legend('Hermite Interpolation', 'Critical Points', 'Provided values');
    % hold off;

    % --- 5) Compute errors between the interpolated values and the provided ones
    % If you have the "exact" values of G(ρ) at the points in E0_values (say, in a
    % variable called G_exact), then you can compare with the interpolation.
    %
    % For example, if E0_values actually holds the exact G(ρ) values, then:
    % interp_vals = p(E0_values);
    % differences = interp_vals - E0_values;
    % max_diff = max(abs(differences));
    % fprintf('Maximum absolute difference between interpolation and provided values: %.4e\n', max_diff);

    % disp(differences);
    % rho_values = linspace(0,1,1000);
    % plot(differences, rho_values, 'g--', 'LineWidth', 1.5);

end

% --------------- Hermite Interpolation for 2 Points -----------------
function [p, dp, coefs] = hermiteInterp(A, B, C, D)
    % A = G(0),  B = G'(0),  C = G(1),  D = G'(1)
    % Compute the cubic coefficients:
    %   p(rho) = a + b*rho + c*rho^2 + d*rho^3
    a = A;
    b = B;
    c = 3*(C - A) - 2*B - D;
    d = -2*(C - A) + B + D;
    coefs = [a, b, c, d];

    % Create function handles for the polynomial and its derivative.
    p  = @(rho) a + b*rho + c*rho.^2 + d*rho.^3;
    dp = @(rho) b + 2*c*rho + 3*d*rho.^2;
end
