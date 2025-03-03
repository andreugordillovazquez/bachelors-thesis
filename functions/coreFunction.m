function [totalEstimatedRunningTime, totalRunningTime, runningTimes] = coreFunction(M, n, SNR, R)
    % coreFunction computes:
    %   1) The total estimated running time (in ms) for all iterations,
    %      based on an empirical factor (in ns) per (n^2 * M^2).
    %   2) The actual running time in seconds (totalRunningTime).
    %   3) The running times for each key section (returned in runningTimes).
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
    %   runningTimes              - Structure array with timing information for each (M,n) iteration

    overallTimer = tic;  % Start overall timing

    fprintf('Simulation Parameters:\n');
    fprintf('  SNR  = %.2f\n', SNR);
    fprintf('  Rate = %.2f\n', R);
    fprintf('  M values: [%s]\n', num2str(M));
    fprintf('  n values: [%s]\n', num2str(n));
    fprintf('--------------------------------------------------------\n');
    
    %% Total Estimated Running Time Calculation
    % Empirical factor from Excel (in nanoseconds per (n^2 * M^2))
    ratio = 112.23;  % ns
    totalEstimatedRunningTime = (ratio * sum(n.^2) * sum(M.^2)) / 1e6;  % convert ns to ms
    fprintf('Total estimated running time over all iterations: %.2f ms\n', totalEstimatedRunningTime);
    fprintf('--------------------------------------------------------\n');

    totalComb = numel(M) * numel(n);
    iterCount = 0;
    
    % Preallocate structure array for timing data
    runningTimes(totalComb) = struct('M_val', [], 'n_val', [], 'M_setup', [], 'n_setup', [], 'Combined', [], 'Optimization', []);
    
    %% Loop for each constellation size M
    for m_idx = 1:numel(M)
        M_val = M(m_idx);
        fprintf('\n================ Processing M = %d (%d of %d) ================\n', ...
            M_val, m_idx, numel(M));
        
        %% M-dependent setup: Generate and normalize the PAM constellation
        tM = tic;
        d = 2;  % Define PAM spacing
        fprintf('-> Generating PAM constellation with M = %d symbols and spacing d = %d...\n', M_val, d);
        X = generatePAMConstellation(M_val, d);
        X = X / sqrt(mean(X.^2));
        fprintf('   Constellation generated and normalized.\n');
        
        % Initialize the probability distribution for constellation symbols (depends only on M)
        Q = repmat(1/M_val, M_val, 1);
        fprintf('-> Probability distribution for constellation symbols initialized.\n');
        tM_elapsed = toc(tM) * 1000;  % Convert to ms
        fprintf('M-dependent setup time: %.6f ms\n', tM_elapsed);
        
        %% Inner Loop: For each number of Gauss-Hermite nodes n
        for n_idx = 1:numel(n)
            iterCount = iterCount + 1;
            n_val = n(n_idx);
            fprintf('\n---------------- Iteration %d of %d ----------------\n', iterCount, totalComb);
            fprintf('Processing for n = %d, M = %d\n', n_val, M_val);
            
            %% n-dependent setup: Compute Gauss-Hermite nodes, weights, and complex nodes matrix
            tN = tic;
            N = n_val;  % Number of Gauss-Hermite nodes
            fprintf('-> Computing Gauss-Hermite nodes and weights with N = %d...\n', N);
            [nodes, weights] = GaussHermite_Locations_Weights(N);
            fprintf('   Gauss-Hermite nodes and weights computed.\n');
            
            % Compute the complex nodes matrix (depends only on n)
            z_matrix = createComplexNodesMatrix(nodes);
            fprintf('-> z_matrix computed for N = %d.\n', N);
            tN_elapsed = toc(tN) * 1000;
            fprintf('n-dependent setup time: %.6f ms\n', tN_elapsed);
            
            %% Combined M & n dependent computations: Create matrices for computation
            tCombined = tic;
            fprintf('-> Creating required matrices for computation...\n');
            % Define the Gaussian function G
            G = @(z) exp(-abs(z).^2) / pi;
            % pi_matrix depends on both M and n
            pi_matrix = createPiMatrix(M_val, N, weights);
            % g_matrix depends on the constellation X (M) and z_matrix (n)
            g_matrix = createGMatrix(X, z_matrix, SNR, G);
            tCombined_elapsed = toc(tCombined) * 1000;
            fprintf('Combined (M & n) computation time: %.6f ms\n', tCombined_elapsed);
            
            %% Optimization routine: Find the Optimal rho
            tOpt = tic;
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
            
            tol = 1e-10;  % Convergence tolerance
            if R > E0P0
                fprintf('   R (%.6f) > derivative at 0 (%.6f): setting optimal rho = 0\n', R, E0P0);
                rho_opt = 0;
                E0_max  = E00;
            elseif R < E0P1
                fprintf('   R (%.6f) < derivative at 1 (%.6f): setting optimal rho = 1\n', R, E0P1);
                rho_opt = 1;
                E0_max  = E01;
            else
                fprintf('-> Running Newton''s method to optimize rho...\n');
                [rho_opt, E0_max] = optimizationNewton(Q, pi_matrix, g_matrix, R, tol, rho_star);
            end
            
            fprintf('   Optimization complete.\n');
            fprintf('   Optimal rho: %.10f, Maximum E0: %.10f\n', rho_opt, E0_max);
            maxER = E0_max - rho_opt*R;
            fprintf('   Maximum E(R): %.10f\n', maxER);
            tOpt_elapsed = toc(tOpt) * 1000;
            fprintf('Optimization routine time: %.6f ms\n', tOpt_elapsed);
            
            %% Store the running times in the structure array
            runningTimes(iterCount).M_val       = M_val;
            runningTimes(iterCount).n_val       = n_val;
            runningTimes(iterCount).M_setup     = tM_elapsed;
            runningTimes(iterCount).n_setup     = tN_elapsed;
            runningTimes(iterCount).Combined    = tCombined_elapsed;
            runningTimes(iterCount).Optimization = tOpt_elapsed;
        end
    end
    
    fprintf('\n==================== END OF SIMULATIONS ====================\n');
    totalRunningTime = toc(overallTimer);  % End overall timing
end