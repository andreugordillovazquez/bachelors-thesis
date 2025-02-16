function [totalEstimatedRunningTime, totalRunningTime] = coreFunction(M, n, SNR, R)
    % coreFunction computes:
    %   1) The total estimated running time (in ms) for all iterations,
    %      based on the empirical factor of 56 ns per (n^2*M^2).
    %   2) The actual running time in seconds (totalRunningTime).
    %   3) The probability of error Pe = 2^(-n_eff * E(R)) for each combination,
    %      where n_eff = log2(M) / R.
    %
    % Inputs:
    %   M   - Constellation sizes (scalar or vector)
    %   n   - Quadrature nodes per real part (scalar or vector)
    %   SNR - Signal-to-noise ratio (scalar)
    %   R   - Transmission rate (scalar)
    %
    % Outputs:
    %   totalEstimatedRunningTime - Total estimated running time (ms) over all iterations
    %   totalRunningTime          - Actual total running time (seconds) measured by tic/toc

    tic;  % Start measuring the actual running time

    fprintf('Simulation Parameters:\n');
    fprintf('  SNR  = %.2f\n', SNR);
    fprintf('  Rate = %.2f\n', R);
    fprintf('  M values: [%s]\n', num2str(M));
    fprintf('  n values: [%s]\n', num2str(n));
    fprintf('--------------------------------------------------------\n');
    
    %% Total Estimated Running Time Calculation
    % Empirical factor from Excel (in nanoseconds per (n^2 * M^2))
    ratio = 56;  % ns
    totalEstimatedRunningTime = (ratio * sum(n.^2) * sum(M.^2)) / 1e6;  % convert ns to ms
    fprintf('Total estimated running time over all iterations: %.2f ms\n', totalEstimatedRunningTime);
    fprintf('--------------------------------------------------------\n');

    totalComb = numel(M) * numel(n);
    iterCount = 0;

    % Create a waitbar to track overall progress
    hWait = waitbar(0, 'Initializing simulations...');

    %% Loop for each constellation size M
    for m_idx = 1:numel(M)
        M_val = M(m_idx);
        fprintf('\n================ Processing M = %d (%d of %d) ================\n', ...
            M_val, m_idx, numel(M));
        
        %% Compute quantities that depend only on M
        % Generate and normalize the PAM constellation once for this M value.
        d = 2;  % Define PAM spacing
        fprintf('-> Generating PAM constellation with M = %d symbols and spacing d = %d...\n', M_val, d);
        X = generatePAMConstellation(M_val, d);
        X = X / sqrt(mean(X.^2));
        fprintf('   Constellation generated and normalized.\n');
        
        % Initialize the probability distribution for constellation symbols (depends only on M)
        Q = repmat(1/M_val, M_val, 1);
        fprintf('-> Probability distribution for constellation symbols initialized.\n');
        
        %% Inner Loop: For each number of Gauss-Hermite nodes n
        for n_idx = 1:numel(n)
            iterCount = iterCount + 1;

            % Update the waitbar
            progress = iterCount / totalComb;
            waitbar(progress, hWait, sprintf('Processing iteration %d of %d', iterCount, totalComb));

            n_val = n(n_idx);
            fprintf('\n---------------- Iteration %d of %d ----------------\n', iterCount, totalComb);
            fprintf('Processing for n = %d, M = %d\n', n_val, M_val);
            
            %% Compute quantities that depend only on n
            % Use n_val as the number of Gauss-Hermite nodes (N)
            N = n_val;
            fprintf('-> Computing Gauss-Hermite nodes and weights with N = %d...\n', N);
            [nodes, weights] = GaussHermite_Locations_Weights(N);
            fprintf('   Gauss-Hermite nodes and weights computed.\n');
            
            % Compute the complex nodes matrix (depends only on n)
            z_matrix = createComplexNodesMatrix(nodes);
            fprintf('-> z_matrix computed for N = %d.\n', N);
            
            %% Combine M- and n-dependent parts
            fprintf('-> Creating required matrices for computation...\n');
            % Define the Gaussian function G
            G = @(z) exp(-abs(z).^2) / pi;
            % pi_matrix depends on both M and n
            pi_matrix = createPiMatrix(M_val, N, weights);
            % g_matrix depends on the constellation X (M) and z_matrix (n)
            g_matrix = createGMatrix(X, z_matrix, SNR, G);
            fprintf('   Required matrices created.\n');
            
            %% Find the Optimal rho
            fprintf('-> Starting optimization for n = %d, M = %d...\n', n_val, M_val);
            % Compute E0 at endpoints (rho = 0 and rho = 1) and their derivatives
            E00  = 0;  % E0 at rho = 0
            E01  = computeEoForRhoExponential(1, Q, pi_matrix, g_matrix);  % E0 at rho = 1
            E0P0 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 0);
            E0P1 = computeFirstDerivativeE0(Q, pi_matrix, g_matrix, 1);
            fprintf('   E0 at rho=0: %.6f, E0 at rho=1: %.6f\n', E00, E01);
            fprintf('   Derivative at rho=0: %.6f, Derivative at rho=1: %.6f\n', E0P0, E0P1);
            
            % Obtain an initial guess for rho using Hermite interpolation
            fprintf('-> Performing Hermite interpolation for an initial guess of rho...\n');
            rho_star = demoHermiteInterpolation(E00, E0P0, E01, E0P1, R);
            fprintf('   Initial guess for rho: %.6f\n', rho_star);
            
            tol = 1e-7;  % Convergence tolerance
            if R > E0P0
                fprintf('   R (%.6f) > derivative at 0 (%.6f): setting optimal rho = 0\n', R, E0P0);
                rho_opt = 0;
                ER_max  = E00;
            elseif R < E0P1
                fprintf('   R (%.6f) < derivative at 1 (%.6f): setting optimal rho = 1\n', R, E0P1);
                rho_opt = 1;
                ER_max  = E01;
            else
                fprintf('-> Running Newton''s method to optimize rho...\n');
                [rho_opt, ER_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star);
            end
            
            fprintf('   Optimization complete.\n');
            fprintf('   Optimal rho: %.6f, Maximum E0: %.6f\n', rho_opt, ER_max);
            
            %% Compute the error probability Pe
            % Calculate effective blocklength using the relation M = 2^(n_eff * R)
            % => n_eff = log2(M) / R.
            n_eff = log2(M_val) / R;
            % Compute Pe = 2^(-n_eff * ER_max)
            Pe = 2^(-n_eff * ER_max);
            fprintf('-> For M = %d and R = %.4f, effective blocklength n_eff = log2(%d)/%.4f = %.4f\n', ...
                M_val, R, M_val, R, n_eff);
            fprintf('   Computed error probability Pe = 2^(-%.4f * %.6f) = %.4e\n', n_eff, ER_max, Pe);
            fprintf('--------------------------------------------------------\n');
        end
    end
    
    % Close the waitbar once all iterations are done
    close(hWait);

    fprintf('\n==================== END OF SIMULATIONS ====================\n');
    totalRunningTime = toc;  % End timing
end